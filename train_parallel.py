import os
import math
import argparse
import random
import logging

import torch
import torch.distributed as dist
import torch.multiprocessing as mp
from data.data_sampler import DistIterSampler

import options.options as option
from utils import util
from data import create_dataloader, create_dataset
from models import create_model
from AverageMeter import *


def init_dist(backend='nccl', **kwargs):
    """initialization for distributed training"""
    if mp.get_start_method(allow_none=True) != 'spawn':
        mp.set_start_method('spawn')
    rank = int(os.environ['RANK'])
    num_gpus = torch.cuda.device_count()

    print("Num GPUs ", num_gpus)
    print("RANK ", rank)

    torch.cuda.set_device(rank % num_gpus)
    dist.init_process_group(backend=backend, **kwargs)


def main():



    ############################################
    #
    #           set options
    #
    ############################################




    parser = argparse.ArgumentParser()
    parser.add_argument('--opt', type=str, help='Path to option YAML file.')
    parser.add_argument('--launcher', choices=['none', 'pytorch'], default='none',
                        help='job launcher')
    parser.add_argument('--local_rank', type=int, default=0)
    args = parser.parse_args()
    opt = option.parse(args.opt, is_train=True)



    ############################################
    #
    #           distributed training settings
    #
    ############################################




    if args.launcher == 'none':  # disabled distributed training
        opt['dist'] = False
        rank = -1
        print('Disabled distributed training.')
    else:
        opt['dist'] = True
        init_dist()
        world_size = torch.distributed.get_world_size()
        rank = torch.distributed.get_rank()

        print("Rank:", rank)
        print("------------------DIST-------------------------")



    ############################################
    #
    #           loading resume state if exists
    #
    ############################################




    if opt['path'].get('resume_state', None):
        # distributed resuming: all load into default GPU
        device_id = torch.cuda.current_device()
        resume_state = torch.load(opt['path']['resume_state'],
                                  map_location=lambda storage, loc: storage.cuda(device_id))
        option.check_resume(opt, resume_state['iter'])  # check resume options
    else:
        resume_state = None




    ############################################
    #
    #           mkdir and loggers
    #
    ############################################




    if rank <= 0:  # normal training (rank -1) OR distributed training (rank 0)
        if resume_state is None:
            util.mkdir_and_rename(opt['path']['experiments_root'])  # rename experiment folder if exists

            util.mkdirs((path for key, path in opt['path'].items() if
                         not key == 'experiments_root' and 'pretrain_model' not in key and 'resume' not in key))

        # config loggers. Before it, the log will not work
        util.setup_logger('base', opt['path']['log'], 'train_' + opt['name'], level=logging.INFO,
                          screen=True, tofile=True)

        util.setup_logger('base_val', opt['path']['log'], 'val_' + opt['name'], level=logging.INFO,
                          screen=True, tofile=True)

        logger = logging.getLogger('base')
        logger_val = logging.getLogger('base_val')

        logger.info(option.dict2str(opt))
        # tensorboard logger
        if opt['use_tb_logger'] and 'debug' not in opt['name']:
            version = float(torch.__version__[0:3])
            if version >= 1.1:  # PyTorch 1.1
                from torch.utils.tensorboard import SummaryWriter
            else:
                logger.info(
                    'You are using PyTorch {}. Tensorboard will use [tensorboardX]'.format(version))
                from tensorboardX import SummaryWriter
            tb_logger = SummaryWriter(log_dir='../tb_logger/' + opt['name'])
    else:
         # config loggers. Before it, the log will not work
        util.setup_logger('base', opt['path']['log'], 'train_', level=logging.INFO, screen=True)
        
        print("set train log")

        util.setup_logger('base_val', opt['path']['log'], 'val_', level=logging.INFO, screen=True)

        print("set val log")

        logger = logging.getLogger('base')

        logger_val = logging.getLogger('base_val')

    # convert to NoneDict, which returns None for missing keys
    opt = option.dict_to_nonedict(opt)

    #### random seed
    seed = opt['train']['manual_seed']
    if seed is None:
        seed = random.randint(1, 10000)
    if rank <= 0:
        logger.info('Random seed: {}'.format(seed))
    util.set_random_seed(seed)

    torch.backends.cudnn.benchmark = True
    # torch.backends.cudnn.deterministic = True



    ############################################
    #
    #           create train and val dataloader
    #
    ############################################
    ####




    # dataset_ratio = 200  # enlarge the size of each epoch, todo: what it is
    dataset_ratio = 1  # enlarge the size of each epoch, todo: what it is
    for phase, dataset_opt in opt['datasets'].items():
        if phase == 'train':
            train_set = create_dataset(dataset_opt)
            train_size = int(math.ceil(len(train_set) / dataset_opt['batch_size']))
            # total_iters = int(opt['train']['niter'])
            # total_epochs = int(math.ceil(total_iters / train_size))

            total_iters = train_size
            total_epochs = int(opt['train']['epoch'])

            if opt['dist']:
                train_sampler = DistIterSampler(train_set, world_size, rank, dataset_ratio)
                # total_epochs = int(math.ceil(total_iters / (train_size * dataset_ratio)))
                total_epochs = int(opt['train']['epoch'])
                if opt['train']['enable'] == False:
                    total_epochs = 1
            else:
                train_sampler = None
            train_loader = create_dataloader(train_set, dataset_opt, opt, train_sampler)
            if rank <= 0:
                logger.info('Number of train images: {:,d}, iters: {:,d}'.format(
                    len(train_set), train_size))
                logger.info('Total epochs needed: {:d} for iters {:,d}'.format(
                    total_epochs, total_iters))
        elif phase == 'val':
            val_set = create_dataset(dataset_opt)
            val_loader = create_dataloader(val_set, dataset_opt, opt, None)
            if rank <= 0:
                logger.info('Number of val images in [{:s}]: {:d}'.format(
                    dataset_opt['name'], len(val_set)))
        else:
            raise NotImplementedError('Phase [{:s}] is not recognized.'.format(phase))

    assert train_loader is not None



    ############################################
    #
    #          create model
    #
    ############################################
    ####



    model = create_model(opt)

    print("Model Created! ")

    #### resume training
    if resume_state:
        logger.info('Resuming training from epoch: {}, iter: {}.'.format(
            resume_state['epoch'], resume_state['iter']))

        start_epoch = resume_state['epoch']
        current_step = resume_state['iter']
        model.resume_training(resume_state)  # handle optimizers and schedulers
    else:
        current_step = 0
        start_epoch = 0
        print("Not Resume Training")


    ############################################
    #
    #          training
    #
    ############################################
    ####


    ####
    logger.info('Start training from epoch: {:d}, iter: {:d}'.format(start_epoch, current_step))
    Avg_train_loss = AverageMeter()  # total
    if (opt['train']['pixel_criterion'] == 'cb+ssim'):
        Avg_train_loss_pix = AverageMeter()
        Avg_train_loss_ssim = AverageMeter()
    elif (opt['train']['pixel_criterion'] == 'cb+ssim+vmaf'):
        Avg_train_loss_pix = AverageMeter()
        Avg_train_loss_ssim = AverageMeter()
        Avg_train_loss_vmaf = AverageMeter()
    elif (opt['train']['pixel_criterion'] == 'ssim'):
        Avg_train_loss_ssim = AverageMeter()
    elif (opt['train']['pixel_criterion'] == 'msssim'):
        Avg_train_loss_msssim = AverageMeter()
    elif (opt['train']['pixel_criterion'] == 'cb+msssim'):
        Avg_train_loss_pix = AverageMeter()
        Avg_train_loss_msssim = AverageMeter()

    saved_total_loss = 10e10
    saved_total_PSNR = -1

    for epoch in range(start_epoch, total_epochs):


        ############################################
        #
        #          Start a new epoch
        #
        ############################################



        # Turn into training mode
        #model = model.train()

        # reset total loss
        Avg_train_loss.reset()
        current_step = 0

        if (opt['train']['pixel_criterion'] == 'cb+ssim'):
            Avg_train_loss_pix.reset()
            Avg_train_loss_ssim.reset()
        elif (opt['train']['pixel_criterion'] == 'cb+ssim+vmaf'):
            Avg_train_loss_pix.reset()
            Avg_train_loss_ssim.reset()
            Avg_train_loss_vmaf.reset()
        elif (opt['train']['pixel_criterion'] == 'ssim'):
            Avg_train_loss_ssim = AverageMeter()
        elif (opt['train']['pixel_criterion'] == 'msssim'):
            Avg_train_loss_msssim = AverageMeter()
        elif (opt['train']['pixel_criterion'] == 'cb+msssim'):
            Avg_train_loss_pix = AverageMeter()
            Avg_train_loss_msssim = AverageMeter()

        if opt['dist']:
            train_sampler.set_epoch(epoch)

        for train_idx, train_data in enumerate(train_loader):

            if 'debug' in opt['name']:

                img_dir = os.path.join(opt['path']['train_images'])
                util.mkdir(img_dir)

                LQ = train_data['LQs']
                GT = train_data['GT']

                GT_img = util.tensor2img(GT)  # uint8

                save_img_path = os.path.join(img_dir, '{:4d}_{:s}.png'.format(train_idx, 'debug_GT'))
                util.save_img(GT_img, save_img_path)

                for i in range(5):
                    LQ_img = util.tensor2img(LQ[0, i, ...])  # uint8
                    save_img_path = os.path.join(img_dir,
                                                 '{:4d}_{:s}_{:1d}.png'.format(train_idx, 'debug_LQ', i))
                    util.save_img(LQ_img, save_img_path)

                if (train_idx >= 3):
                    break

            if opt['train']['enable'] == False:
                message_train_loss = 'None'
                break

            current_step += 1
            if current_step > total_iters:
                print("Total Iteration Reached !")
                break
            #### update learning rate
            if opt['train']['lr_scheme'] == 'ReduceLROnPlateau':
                pass
            else:
                model.update_learning_rate(current_step, warmup_iter=opt['train']['warmup_iter'])

            #### training
            model.feed_data(train_data)

            # if opt['train']['lr_scheme'] == 'ReduceLROnPlateau':
            #    model.optimize_parameters_without_schudlue(current_step)
            # else:
            model.optimize_parameters(current_step)

            if (opt['train']['pixel_criterion'] == 'cb+ssim'):
                Avg_train_loss.update(model.log_dict['total_loss'], 1)
                Avg_train_loss_pix.update(model.log_dict['l_pix'], 1)
                Avg_train_loss_ssim.update(model.log_dict['ssim_loss'], 1)
            elif (opt['train']['pixel_criterion'] == 'cb+ssim+vmaf'):
                Avg_train_loss.update(model.log_dict['total_loss'], 1)
                Avg_train_loss_pix.update(model.log_dict['l_pix'], 1)
                Avg_train_loss_ssim.update(model.log_dict['ssim_loss'], 1)
                Avg_train_loss_vmaf.update(model.log_dict['vmaf_loss'], 1)
            elif (opt['train']['pixel_criterion'] == 'ssim'):
                Avg_train_loss.update(model.log_dict['total_loss'], 1)
                Avg_train_loss_ssim.update(model.log_dict['ssim_loss'], 1)
            elif (opt['train']['pixel_criterion'] == 'msssim'):
                Avg_train_loss.update(model.log_dict['total_loss'], 1)
                Avg_train_loss_msssim.update(model.log_dict['msssim_loss'], 1)
            elif (opt['train']['pixel_criterion'] == 'cb+msssim'):
                Avg_train_loss.update(model.log_dict['total_loss'], 1)
                Avg_train_loss_pix.update(model.log_dict['l_pix'], 1)
                Avg_train_loss_msssim.update(model.log_dict['msssim_loss'], 1)
            else:
                Avg_train_loss.update(model.log_dict['l_pix'], 1)

            # add total train loss
            if (opt['train']['pixel_criterion'] == 'cb+ssim'):
                message_train_loss = ' pix_avg_loss: {:.4e}'.format(Avg_train_loss_pix.avg)
                message_train_loss += ' ssim_avg_loss: {:.4e}'.format(Avg_train_loss_ssim.avg)
                message_train_loss += ' total_avg_loss: {:.4e}'.format(Avg_train_loss.avg)
            elif (opt['train']['pixel_criterion'] == 'cb+ssim+vmaf'):
                message_train_loss = ' pix_avg_loss: {:.4e}'.format(Avg_train_loss_pix.avg)
                message_train_loss += ' ssim_avg_loss: {:.4e}'.format(Avg_train_loss_ssim.avg)
                message_train_loss += ' vmaf_avg_loss: {:.4e}'.format(Avg_train_loss_vmaf.avg)
                message_train_loss += ' total_avg_loss: {:.4e}'.format(Avg_train_loss.avg)
            elif (opt['train']['pixel_criterion'] == 'ssim'):
                message_train_loss = ' ssim_avg_loss: {:.4e}'.format(Avg_train_loss_ssim.avg)
                message_train_loss += ' total_avg_loss: {:.4e}'.format(Avg_train_loss.avg)
            elif (opt['train']['pixel_criterion'] == 'msssim'):
                message_train_loss = ' msssim_avg_loss: {:.4e}'.format(Avg_train_loss_msssim.avg)
                message_train_loss += ' total_avg_loss: {:.4e}'.format(Avg_train_loss.avg)
            elif (opt['train']['pixel_criterion'] == 'cb+msssim'):
                message_train_loss = ' pix_avg_loss: {:.4e}'.format(Avg_train_loss_pix.avg)
                message_train_loss += ' msssim_avg_loss: {:.4e}'.format(Avg_train_loss_msssim.avg)
                message_train_loss += ' total_avg_loss: {:.4e}'.format(Avg_train_loss.avg)
            else:
                message_train_loss = ' train_avg_loss: {:.4e}'.format(Avg_train_loss.avg)

            #### log
            if current_step % opt['logger']['print_freq'] == 0:
                logs = model.get_current_log()
                message = '[epoch:{:3d}, iter:{:8,d}, lr:('.format(epoch, current_step)
                for v in model.get_current_learning_rate():
                    message += '{:.3e},'.format(v)
                message += ')] '
                for k, v in logs.items():
                    message += '{:s}: {:.4e} '.format(k, v)
                    # tensorboard logger
                    if opt['use_tb_logger'] and 'debug' not in opt['name']:
                        if rank <= 0:
                            tb_logger.add_scalar(k, v, current_step)

                message += message_train_loss

                if rank <= 0:
                    logger.info(message)



        ############################################
        #
        #        end of one epoch, save epoch model
        #
        ############################################



        #### save models and training states
        # if current_step % opt['logger']['save_checkpoint_freq'] == 0:
        #     if rank <= 0:
        #         logger.info('Saving models and training states.')
        #         model.save(current_step)
        #         model.save('latest')
        #         # model.save_training_state(epoch, current_step)
        #         # todo delete previous weights
        #         previous_step = current_step - opt['logger']['save_checkpoint_freq']
        #         save_filename = '{}_{}.pth'.format(previous_step, 'G')
        #         save_path = os.path.join(opt['path']['models'], save_filename)
        #         if os.path.exists(save_path):
        #             os.remove(save_path)

        if epoch == 1:
            save_filename = '{:04d}_{}.pth'.format(0, 'G')
            save_path = os.path.join(opt['path']['models'], save_filename)
            if os.path.exists(save_path):
                os.remove(save_path)

        save_filename = '{:04d}_{}.pth'.format(epoch-1, 'G')
        save_path = os.path.join(opt['path']['models'], save_filename)
        if os.path.exists(save_path):
            os.remove(save_path)

        if rank <= 0:
            logger.info('Saving models and training states.')
            save_filename = '{:04d}'.format(epoch)
            model.save(save_filename)
            # model.save('latest')
            # model.save_training_state(epoch, current_step)



        ############################################
        #
        #          end of one epoch, do validation
        #
        ############################################



        #### validation
        #if opt['datasets'].get('val', None) and current_step % opt['train']['val_freq'] == 0:
        if opt['datasets'].get('val', None) :
            if opt['model'] in ['sr', 'srgan'] and rank <= 0:  # image restoration validation
                # does not support multi-GPU validation
                pbar = util.ProgressBar(len(val_loader))
                avg_psnr = 0.
                idx = 0
                for val_data in val_loader:
                    idx += 1
                    img_name = os.path.splitext(os.path.basename(val_data['LQ_path'][0]))[0]
                    img_dir = os.path.join(opt['path']['val_images'], img_name)
                    util.mkdir(img_dir)

                    model.feed_data(val_data)
                    model.test()
                    visuals = model.get_current_visuals()
                    sr_img = util.tensor2img(visuals['rlt'])  # uint8
                    gt_img = util.tensor2img(visuals['GT'])  # uint8

                    # Save SR images for reference
                    save_img_path = os.path.join(img_dir,
                                                 '{:s}_{:d}.png'.format(img_name, current_step))
                    #util.save_img(sr_img, save_img_path)

                    # calculate PSNR
                    sr_img, gt_img = util.crop_border([sr_img, gt_img], opt['scale'])
                    avg_psnr += util.calculate_psnr(sr_img, gt_img)
                    pbar.update('Test {}'.format(img_name))

                avg_psnr = avg_psnr / idx

                # log
                logger.info('# Validation # PSNR: {:.4e}'.format(avg_psnr))
                # tensorboard logger
                if opt['use_tb_logger'] and 'debug' not in opt['name']:
                    tb_logger.add_scalar('psnr', avg_psnr, current_step)
            else:  # video restoration validation
                if opt['dist']:
                    # todo : multi-GPU testing
                    psnr_rlt = {}  # with border and center frames
                    psnr_rlt_avg = {}
                    psnr_total_avg = 0.

                    ssim_rlt = {}  # with border and center frames
                    ssim_rlt_avg = {}
                    ssim_total_avg = 0.

                    val_loss_rlt = {}
                    val_loss_rlt_avg ={}
                    val_loss_total_avg = 0.

                    if rank == 0:
                        pbar = util.ProgressBar(len(val_set))

                    for idx in range(rank, len(val_set), world_size):

                        print('idx', idx)

                        if 'debug' in opt['name']:
                            if (idx >= 3):
                                break

                        val_data = val_set[idx]
                        val_data['LQs'].unsqueeze_(0)
                        val_data['GT'].unsqueeze_(0)
                        folder = val_data['folder']
                        idx_d, max_idx = val_data['idx'].split('/')
                        idx_d, max_idx = int(idx_d), int(max_idx)

                        if psnr_rlt.get(folder, None) is None:
                            psnr_rlt[folder] = torch.zeros(max_idx, dtype=torch.float32,
                                                           device='cuda')

                        if ssim_rlt.get(folder, None) is None:
                            ssim_rlt[folder] = torch.zeros(max_idx, dtype=torch.float32,
                                                           device='cuda')

                        if val_loss_rlt.get(folder, None) is None:
                            val_loss_rlt[folder] = torch.zeros(max_idx, dtype=torch.float32,
                                                           device='cuda')

                        # tmp = torch.zeros(max_idx, dtype=torch.float32, device='cuda')
                        model.feed_data(val_data)
                        # model.test()
                        # model.test_stitch()

                        if opt['stitch'] == True:
                            model.test_stitch()
                        else:
                            model.test() # large GPU memory

                        # visuals = model.get_current_visuals()
                        visuals = model.get_current_visuals(save=True, name='{}_{}'.format(folder, idx),
                                                            save_path=opt['path']['val_images'])

                        rlt_img = util.tensor2img(visuals['rlt'])  # uint8
                        gt_img = util.tensor2img(visuals['GT'])  # uint8


                        # calculate PSNR
                        psnr = util.calculate_psnr(rlt_img, gt_img)
                        psnr_rlt[folder][idx_d] = psnr

                        # calculate SSIM
                        ssim = util.calculate_ssim(rlt_img, gt_img)
                        ssim_rlt[folder][idx_d] = ssim

                        # calculate Val loss
                        val_loss = model.get_loss()
                        val_loss_rlt[folder][idx_d] = val_loss

                        logger.info('{}_{:02d} PSNR: {:.4f}, SSIM: {:.4f}'.format(folder, idx, psnr, ssim))

                        if rank == 0:
                            for _ in range(world_size):
                                pbar.update('Test {} - {}/{}'.format(folder, idx_d, max_idx))

                    # # collect data
                    for _, v in psnr_rlt.items():
                        dist.reduce(v, 0)

                    for _, v in ssim_rlt.items():
                        dist.reduce(v, 0)

                    for _, v in val_loss_rlt.items():
                        dist.reduce(v, 0)

                    dist.barrier()

                    if rank == 0:
                        psnr_rlt_avg = {}
                        psnr_total_avg = 0.
                        for k, v in psnr_rlt.items():
                            psnr_rlt_avg[k] = torch.mean(v).cpu().item()
                            psnr_total_avg += psnr_rlt_avg[k]
                        psnr_total_avg /= len(psnr_rlt)
                        log_s = '# Validation # PSNR: {:.4e}:'.format(psnr_total_avg)
                        for k, v in psnr_rlt_avg.items():
                            log_s += ' {}: {:.4e}'.format(k, v)
                        logger.info(log_s)

                        # ssim
                        ssim_rlt_avg = {}
                        ssim_total_avg = 0.
                        for k, v in ssim_rlt.items():
                            ssim_rlt_avg[k] = torch.mean(v).cpu().item()
                            ssim_total_avg += ssim_rlt_avg[k]
                        ssim_total_avg /= len(ssim_rlt)
                        log_s = '# Validation # PSNR: {:.4e}:'.format(ssim_total_avg)
                        for k, v in ssim_rlt_avg.items():
                            log_s += ' {}: {:.4e}'.format(k, v)
                        logger.info(log_s)

                        # added
                        val_loss_rlt_avg = {}
                        val_loss_total_avg = 0.
                        for k, v in val_loss_rlt.items():
                            val_loss_rlt_avg[k] = torch.mean(v).cpu().item()
                            val_loss_total_avg += val_loss_rlt_avg[k]
                        val_loss_total_avg /= len(val_loss_rlt)
                        log_l = '# Validation # Loss: {:.4e}:'.format(val_loss_total_avg)
                        for k, v in val_loss_rlt_avg.items():
                            log_l += ' {}: {:.4e}'.format(k, v)
                        logger.info(log_l)

                        message = ''
                        for v in model.get_current_learning_rate():
                            message += '{:.5e}'.format(v)

                        logger_val.info(
                            'Epoch {:02d}, LR {:s}, PSNR {:.4f}, SSIM {:.4f} Train {:s}, Val Total Loss {:.4e}'.format(
                                epoch, message, psnr_total_avg, ssim_total_avg, message_train_loss, val_loss_total_avg
                            ))



                        if opt['use_tb_logger'] and 'debug' not in opt['name']:
                            tb_logger.add_scalar('psnr_avg', psnr_total_avg, current_step)
                            for k, v in psnr_rlt_avg.items():
                                tb_logger.add_scalar(k, v, current_step)
                            # add val loss
                            tb_logger.add_scalar('val_loss_avg', val_loss_total_avg, current_step)
                            for k, v in val_loss_rlt_avg.items():
                                tb_logger.add_scalar(k, v, current_step)


                else:  # Todo: our function One GPU
                    pbar = util.ProgressBar(len(val_loader))
                    psnr_rlt = {}  # with border and center frames
                    psnr_rlt_avg = {}
                    psnr_total_avg = 0.

                    ssim_rlt = {}  # with border and center frames
                    ssim_rlt_avg = {}
                    ssim_total_avg = 0.

                    val_loss_rlt = {}
                    val_loss_rlt_avg = {}
                    val_loss_total_avg = 0.

                    for val_inx, val_data in enumerate(val_loader):

                        if 'debug' in opt['name']:
                            if (val_inx >= 5):
                                break

                        folder = val_data['folder'][0]
                        # idx_d = val_data['idx'].item()
                        idx_d = val_data['idx']
                        # border = val_data['border'].item()
                        if psnr_rlt.get(folder, None) is None:
                            psnr_rlt[folder] = []

                        if ssim_rlt.get(folder, None) is None:
                            ssim_rlt[folder] = []

                        if val_loss_rlt.get(folder, None) is None:
                            val_loss_rlt[folder] = []

                        # process the black blank [B N C H W]

                        print(val_data['LQs'].size())

                        H_S = val_data['LQs'].size(3)  # 540
                        W_S = val_data['LQs'].size(4)  # 960

                        print(H_S)
                        print(W_S)

                        blank_1_S = 0
                        blank_2_S = 0

                        print(val_data['LQs'][0,2,0,:,:].size())

                        for i in range(H_S):
                            if not sum(val_data['LQs'][0,2,0,i,:]) == 0:
                                blank_1_S = i - 1
                                # assert not sum(data_S[:, :, 0][i+1]) == 0
                                break

                        for i in range(H_S):
                            if not sum(val_data['LQs'][0,2,0,:,H_S - i - 1]) == 0:
                                blank_2_S = (H_S - 1) - i - 1
                                # assert not sum(data_S[:, :, 0][blank_2_S-1]) == 0
                                break
                        print('LQ :', blank_1_S, blank_2_S)

                        if blank_1_S == -1:
                            print('LQ has no blank')
                            blank_1_S = 0
                            blank_2_S = H_S

                        # val_data['LQs'] = val_data['LQs'][:,:,:,blank_1_S:blank_2_S,:]

                        print("LQ",val_data['LQs'].size())

                        # end of process the black blank

                        model.feed_data(val_data)

                        if opt['stitch'] == True:
                            model.test_stitch()
                        else:
                            model.test()  # large GPU memory

                        # process blank

                        blank_1_L = blank_1_S << 2
                        blank_2_L = blank_2_S << 2
                        print(blank_1_L, blank_2_L)

                        print(model.fake_H.size())

                        if not blank_1_S == 0:
                            # model.fake_H = model.fake_H[:,:,blank_1_L:blank_2_L,:]
                            model.fake_H[:, :,0:blank_1_L, :] = 0
                            model.fake_H[:, :,blank_2_L:H_S, :] = 0

                        # end of # process blank

                        visuals = model.get_current_visuals(save=True,
                                                            name='{}_{:02d}'.format(folder, val_inx),
                                                            save_path=opt['path']['val_images'])

                        rlt_img = util.tensor2img(visuals['rlt'])  # uint8
                        gt_img = util.tensor2img(visuals['GT'])  # uint8




                        # calculate PSNR
                        psnr = util.calculate_psnr(rlt_img, gt_img)
                        psnr_rlt[folder].append(psnr)

                        # calculate SSIM
                        ssim = util.calculate_ssim(rlt_img, gt_img)
                        ssim_rlt[folder].append(ssim)

                        # val loss
                        val_loss = model.get_loss()
                        val_loss_rlt[folder].append(val_loss.item())

                        logger.info('{}_{:02d} PSNR: {:.4f}, SSIM: {:.4f}'.format(folder, val_inx, psnr, ssim))

                        pbar.update('Test {} - {}'.format(folder, idx_d))

                    # average PSNR
                    for k, v in psnr_rlt.items():
                        psnr_rlt_avg[k] = sum(v) / len(v)
                        psnr_total_avg += psnr_rlt_avg[k]
                    psnr_total_avg /= len(psnr_rlt)
                    log_s = '# Validation # PSNR: {:.4e}:'.format(psnr_total_avg)
                    for k, v in psnr_rlt_avg.items():
                        log_s += ' {}: {:.4e}'.format(k, v)
                    logger.info(log_s)

                    # average SSIM
                    for k, v in ssim_rlt.items():
                        ssim_rlt_avg[k] = sum(v) / len(v)
                        ssim_total_avg += ssim_rlt_avg[k]
                    ssim_total_avg /= len(ssim_rlt)
                    log_s = '# Validation # SSIM: {:.4e}:'.format(ssim_total_avg)
                    for k, v in ssim_rlt_avg.items():
                        log_s += ' {}: {:.4e}'.format(k, v)
                    logger.info(log_s)

                    # average VMAF


                    # average Val LOSS
                    for k, v in val_loss_rlt.items():
                        val_loss_rlt_avg[k] = sum(v) / len(v)
                        val_loss_total_avg += val_loss_rlt_avg[k]
                    val_loss_total_avg /= len(val_loss_rlt)
                    log_l = '# Validation # Loss: {:.4e}:'.format(val_loss_total_avg)
                    for k, v in val_loss_rlt_avg.items():
                        log_l += ' {}: {:.4e}'.format(k, v)
                    logger.info(log_l)

                    # toal validation log

                    message = ''
                    for v in model.get_current_learning_rate():
                        message += '{:.5e}'.format(v)

                    logger_val.info('Epoch {:02d}, LR {:s}, PSNR {:.4f}, SSIM {:.4f} Train {:s}, Val Total Loss {:.4e}'.format(
                        epoch, message, psnr_total_avg, ssim_total_avg ,message_train_loss, val_loss_total_avg
                    ))

                    # end add

                    if opt['use_tb_logger'] and 'debug' not in opt['name']:
                        tb_logger.add_scalar('psnr_avg', psnr_total_avg, current_step)
                        for k, v in psnr_rlt_avg.items():
                            tb_logger.add_scalar(k, v, current_step)
                        # tb_logger.add_scalar('ssim_avg', ssim_total_avg, current_step)
                        # for k, v in ssim_rlt_avg.items():
                        #     tb_logger.add_scalar(k, v, current_step)
                        # add val loss
                        tb_logger.add_scalar('val_loss_avg', val_loss_total_avg, current_step)
                        for k, v in val_loss_rlt_avg.items():
                            tb_logger.add_scalar(k, v, current_step)



            ############################################
            #
            #          end of validation, save model
            #
            ############################################
            #
            logger.info("Finished an epoch, Check and Save the model weights")
            # we check the validation loss instead of training loss. OK~
            if saved_total_loss >= val_loss_total_avg:
                saved_total_loss = val_loss_total_avg
                #torch.save(model.state_dict(), args.save_path + "/best" + ".pth")
                model.save('best')
                logger.info("Best Weights updated for decreased validation loss")

            else:
                logger.info("Weights Not updated for undecreased validation loss")

            if saved_total_PSNR <= psnr_total_avg:
                saved_total_PSNR = psnr_total_avg
                model.save('bestPSNR')
                logger.info("Best Weights updated for increased validation PSNR")

            else:
                logger.info("Weights Not updated for unincreased validation PSNR")



        ############################################
        #
        #          end of one epoch, schedule LR
        #
        ############################################



        # add scheduler  todo
        if opt['train']['lr_scheme'] == 'ReduceLROnPlateau':
            for scheduler in model.schedulers:
                # scheduler.step(val_loss_total_avg)
                scheduler.step(val_loss_total_avg)

    if rank <= 0:
        logger.info('Saving the final model.')
        model.save('last')
        logger.info('End of training.')
        tb_logger.close()


if __name__ == '__main__':
    main()