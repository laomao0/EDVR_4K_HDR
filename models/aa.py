
# image = train_data['GT'].detach()[0].float().cpu()

image = util.tensor2img(train_data['GT'].detach()[0].float().cpu())

cv2.imwrite(osp.join("./", '{}.png'.format("debug")), util.tensor2img(train_data['GT'].detach()[0].float().cpu()))
