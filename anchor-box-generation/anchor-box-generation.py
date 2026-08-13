def generate_anchors(feature_size, image_size, scales, aspect_ratios):
    """
    Generate anchor boxes for object detection.
    """
    # Write code here
    stride = image_size/feature_size

    arr = []
    
    for i in range(feature_size):
        for j in range(feature_size):
            cx = (j+1/2)*stride
            cy = (i+1/2)*stride

            for s in scales:
                for r in aspect_ratios:
                    w = s*((r)**(1/2)) 
                    h = s / ((r) **(1/2))

                    x1 = cx - w / 2
                    y1 = cy - h / 2
                    x2 = cx + w / 2
                    y2 = cy + h / 2

                    arr.append([x1, y1, x2, y2])

    return arr