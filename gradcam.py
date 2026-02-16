from pytorch_grad_cam import GradCAM
from pytorch_grad_cam.utils.image import show_cam_on_image
from pytorch_grad_cam.utils.model_targets import SemanticSegmentationTarget

def gradcam(model, input_tensor, img, mask):
    target_layers = [model.multi_task_MaxViT.backbone.stages[3]]
    targets = [SemanticSegmentationTarget(category=0, mask=mask)]

    cam = GradCAM(model, target_layers)
    grayscale_cam = cam(input_tensor, targets)[0, :]
    cam_image = show_cam_on_image(img, grayscale_cam, use_rgb=True)

    return cam_image