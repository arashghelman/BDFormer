import numpy as np
from tqdm import tqdm
import torch
from torch.cuda.amp import autocast as autocast
from utils import save_imgs_multitask, Caculate_multi_task_loss, evaluate_segmentation_metrics, evaluate_classification_metrics, create_results_info, save_msk_pred, save_msk_contour
import torch.nn as nn

def train_one_epoch_multi(
        data_loader,
        class_weights,
        model,
        optimizer,
        scheduler,
        epoch,
        logger,
        config):
    loss_list = []

    # switch to train mode
    model.train()

    coefficient_generator = nn.Sigmoid()
    alpha = coefficient_generator(torch.tensor(epoch / 100 * 20 - 10))

    for iter, data in enumerate(data_loader):
        optimizer.zero_grad()

        target_img, target_seg, target_contour, target_class = data
        target_img = target_img.float().cuda(non_blocking=True)
        target_seg = target_seg.float().cuda(non_blocking=True)
        target_contour = target_contour.float().cuda(non_blocking=True)
        target_contour = target_contour.squeeze(dim=1)
        target_class = target_class.long().cuda(non_blocking=True)

        pred_seg, pred_contour, pred_class = model(target_img)
        loss = Caculate_multi_task_loss(
            pred_seg,
            pred_contour,
            pred_class,
            target_seg,
            target_contour,
            target_class,
            class_weights,
            alpha)
        loss.backward()
        optimizer.step()

        loss_list.append(loss.item())

        now_lr = optimizer.state_dict()['param_groups'][0]['lr']
        if iter % config.print_interval == 0:
            log_info = f'train: epoch {epoch}, iter:{iter}, loss: {np.mean(loss_list):.4f}, lr: {now_lr}'
            print(log_info)
            logger.info(log_info)

    scheduler.step()

def val_one_epoch_multi(
        data_loader,
        class_weights,
        model,
        epoch,
        logger,
        config):
    pred_masks = []
    target_masks = []
    pred_classes = []
    target_classes = []
    loss_list = []
    alpha = 1

    # switch to evaluate mode
    model.eval()

    with torch.no_grad():
        for data in tqdm(data_loader, ncols=70):
            target_img, target_seg, target_contour, target_class = data
            target_img = target_img.float().cuda(non_blocking=True)
            target_seg = target_seg.float().cuda(non_blocking=True)
            target_contour = target_contour.float().cuda(non_blocking=True)
            target_contour = target_contour.squeeze(dim=1)
            target_class = target_class.long().cuda(non_blocking=True)

            pred_seg, pred_contour, pred_class = model(target_img)

            loss = Caculate_multi_task_loss(
                pred_seg, 
                pred_contour,
                pred_class,
                target_seg, 
                target_contour,
                target_class,
                class_weights,
                alpha)
            loss_list.append(loss.item())

            target_masks.append(target_seg.squeeze(1).cpu().detach().numpy())

            if type(pred_seg) is tuple: pred_seg = pred_seg[0]
            pred_seg = pred_seg.squeeze(1).cpu().detach().numpy()
            pred_masks.append(pred_seg)

            target_classes.append(target_class.cpu().detach().numpy())
            pred_classes.append(pred_class.cpu().detach().numpy())

    if epoch % config.val_interval == 0:
        segmentation_metrics = evaluate_segmentation_metrics(target_masks, pred_masks, config.threshold)
        classification_metrics = evaluate_classification_metrics(target_classes, pred_classes, config.class_map)
        
        results_info = create_results_info(segmentation_metrics, classification_metrics)
        log_info = (
            f"\nval epoch: {epoch}, loss: {np.mean(loss_list):.4f}\n"
            f"{results_info}"
        )

        print(log_info)
        logger.info(log_info)
    else:
        log_info = f'val epoch: {epoch}, loss: {np.mean(loss_list):.4f}'
        print(log_info)
        logger.info(log_info)

    return np.mean(loss_list)

def test_one_epoch_multi(
        data_loader,
        model,
        criterion,
        logger,
        config,
        test_data_name=None):
    pred_masks = []
    target_masks = []
    target_classes = []
    pred_classes = []
    loss_list = []

    # switch to evaluate mode
    model.eval()

    for i, data in enumerate(tqdm(data_loader, ncols=70)):
        target_img, target_seg, target_contour, target_class = data
        target_img = target_img.float().cuda(non_blocking=True)
        target_seg = target_seg.float().cuda(non_blocking=True)
        target_contour = target_contour.float().cuda(non_blocking=True)

        with torch.no_grad():
            pred_seg, pred_contour, pred_class = model(target_img)
            loss = criterion(pred_seg, target_seg)
            loss_list.append(loss.item())

            target_seg = target_seg.squeeze(1).cpu().detach().numpy()
            target_masks.append(target_seg)

        target_img = target_img.squeeze(0).permute(1,2,0).detach().cpu().numpy()
        target_img = target_img / 255. if target_img.max() > 1.1 else target_img

        if type(pred_seg) is tuple: pred_seg = pred_seg[0]
        pred_seg = pred_seg.squeeze(1).cpu().detach().numpy()
        pred_masks.append(pred_seg)

        pred_contour = torch.argmax(pred_contour, dim=1).squeeze(0).cpu().detach().numpy()

        target_classes.append(target_class.cpu().detach().numpy())
        pred_classes.append(pred_class.cpu().detach().numpy())

        # if i % config.save_interval == 0:
        #     save_imgs_multitask(
        #         target_img, 
        #         target_seg, 
        #         pred_seg, 
        #         pred_contour, 
        #         i, 
        #         config.work_dir,
        #         config.datasets, 
        #         config.threshold,
        #         test_data_name=test_data_name)

        #     save_msk_pred(pred_seg, i, config.work_dir, config.threshold)
        #     save_msk_contour(pred_contour, i, config.work_dir)

    segmentation_metrics = evaluate_segmentation_metrics(target_masks, pred_masks, config.threshold)
    classification_metrics = evaluate_classification_metrics(target_classes, pred_classes, config.class_map)

    if test_data_name is not None:
        log_info = f'test_datasets_name: {test_data_name}'
        print(log_info)
        logger.info(log_info)

    results_info = create_results_info(segmentation_metrics, classification_metrics)
    log_info = (
        f"test of best model, loss: {np.mean(loss_list):.4f}"
        f"{results_info}"
    )
    
    print(log_info)
    logger.info(log_info)

    return np.mean(loss_list)