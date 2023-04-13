import yolov5
# 'fcakyon/yolov5s-v7.0'
def object_detect(model_name = 'fcakyon/yolov5s-v7.0'):
    # load model
    model = yolov5.load(model_name)
    
    # set model parameters
    model.conf = 0.25  # NMS confidence threshold
    model.iou = 0.45  # NMS IoU threshold
    model.agnostic = False  # NMS class-agnostic
    model.multi_label = False  # NMS multiple labels per box
    model.max_det = 1000  # maximum number of detections per image

    # set image
    img = 'https://media.istockphoto.com/id/618546412/photo/pizza-and-hamburger-on-wooden-background.jpg?s=612x612&w=0&k=20&c=vunadZByjJhV_S7DqjlmpM9hQMk_Y7dVrNt9pYCu_Ww='

    # perform inference
    results = model(img)

    # inference with larger input size
    results = model(img, size=640)

    # inference with test time augmentation
    results = model(img, augment=True)

    # parse results
    predictions = results.pred[0]
    boxes = predictions[:, :4] # x1, y1, x2, y2
    scores = predictions[:, 4]
    categories = predictions[:, 5]

    # show detection bounding boxes on image
    results.show()

    # save results into "results/" folder
    results.save(save_dir='results/')
