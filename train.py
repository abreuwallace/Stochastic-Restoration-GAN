import logging
import time
import hydra
import torch

from src.data.dataset import AudioFolder
from src.models.generator import Generator
from src.models.discriminator import Discriminator, discriminator_loss, generator_loss, gradient_penalty
from src.models.utils import model_size_log, LogProgress, init_weights, bold

from torch.utils.data import DataLoader

logger = logging.getLogger('training')

def Trainer(args):
    if args.logging == False:
        logger.disabled = True

    G = Generator(args.experiment.generator.enc_params, args.experiment.generator.int_params, args.experiment.generator.dec_params)
    D = Discriminator(args.experiment.discriminator.enc_params, args.experiment.discriminator.res_params, args.experiment.discriminator.logit_params)
    G.apply(init_weights)
    D.apply(init_weights)

    train_dataset = AudioFolder(args.dset.datapath, 
                                args.dset.suffix,
                                args.dset.train_pattern,
                                args.experiment.fs,
                                args.experiment.seg_len,
                                args.experiment.overlap,
                                args.experiment.stft,
                                args.experiment.stft_scaling, 
                                args.experiment.win_len,
                                args.experiment.hop_len, 
                                args.experiment.n_fft,
                                args.experiment.complex_as_channels)
    
    # should be adapted to ddp loader
    train_loader = DataLoader(train_dataset, batch_size=args.experiment.batch_size, shuffle=True, num_workers=args.num_workers)

    if torch.cuda.is_available() and args.device == 'cuda':
        logger.info('Using CUDA')
        G = G.to(args.device)
        D = D.to(args.device)
    else:
        logger.info('No Graphics Card available. Defaulting to CPU')

    if args.experiment.optim == "adam":
        optimizer_g = torch.optim.Adam(G.parameters(), lr=args.experiment.lr, betas=(args.experiment.beta1, args.experiment.beta2))
        optimizer_d = torch.optim.Adam(D.parameters(), lr=args.experiment.lr, betas=(args.experiment.beta1, args.experiment.beta2))

    logger.info('-' * 70)
    logger.info("Trainable Params:")
    model_size_log(logger, G)
    model_size_log(logger, D)

    logger.info('-' * 70)
    logger.info("Training...")

    losses = {"disc_loss": 0, 
              "gen_loss": 0}

    for epoch in range(args.experiment.epochs):
        logprog = LogProgress(logger, train_loader, updates=args.num_prints, name="train")

        for i, data in enumerate(logprog):
            start = time.time()
            x, y = [x.to(args.device) for x in data]
            z = torch.randn(args.experiment.batch_size, 64, 1, 1).to(args.device)
            y_hat = G(x, z)
            y_sliced = y[:, :, :, 62:-62] #remove 62 frames from begin and end
            D_real = D(y_sliced)
            D_fake = D(y_hat.detach())

            optimizer_d.zero_grad()
            loss_disc = discriminator_loss(D_real, D_fake) + gradient_penalty(D, y_sliced, y_hat.detach(), args.device, args.experiment.grad_penalty_weight)
            loss_disc.backward()
            optimizer_d.step()
            losses["disc_loss"] += loss_disc 

            D_fake = D(y_hat)
            optimizer_g.zero_grad()
            loss_gen = generator_loss(D_fake)
            loss_gen.backward()
            optimizer_g.step()
            losses["gen_loss"] += loss_gen
            logger_msg = f'Train Summary | End of Epoch {epoch + 1} | Time {time.time() - start:.2f}s | ' \
                         + ' | '.join([f'{k} Loss {v:.5f}' for k, v in losses.items()])
            logger.info(bold(logger_msg))

            
@hydra.main(version_base="1.2", config_path="conf", config_name="conf")  
def main(args):
    try:
        Trainer(args)
    except Exception:
        logger.exception("Couldn't run Trainer")

if __name__ == "__main__":
    main()