import cv2
import numpy

def get_bound_image(image, angle):
    height, width = image.shape[:2]
    center = (width // 2, height // 2)

    affine = cv2.getRotationMatrix2D(center, angle, 1.0)
    cosofRotationMatrix = numpy.abs(affine[0][0])
    sinofRotationMatrix = numpy.abs(affine[0][1])

    BoundingImageWidth = int((height * sinofRotationMatrix) + (width * cosofRotationMatrix))
    BoundingImageHeight = int((height * cosofRotationMatrix) + (width * sinofRotationMatrix))

    affine[0][2] += (BoundingImageWidth/2) - center[0]
    affine[1][2] += (BoundingImageHeight/2) - center[1]
    bound_image = cv2.warpAffine(image, affine, (BoundingImageWidth, BoundingImageHeight))
    return bound_image

def get_inner_square_image(bound_image, image, angle):
    height, width = image.shape[:2]
    center = (bound_image.shape[1] // 2, bound_image.shape[0] // 2)
    
    cosofRotationMatrix = numpy.abs(numpy.cos(angle*numpy.pi/180))
    sinofRotationMatrix = numpy.abs(numpy.sin(angle*numpy.pi/180))
    
    squareImageHeight = int(numpy.min([width, height]) / (sinofRotationMatrix + cosofRotationMatrix))
    squareImageWidth = int(numpy.min([width, height]) / (sinofRotationMatrix + cosofRotationMatrix))
    return bound_image[center[1]-(squareImageHeight//2):center[1]+(squareImageHeight//2), center[0]-(squareImageWidth//2):center[0]+(squareImageWidth//2)]

def get_inner_rectangle_image(bound_image, image, angle):
    height, width = image.shape[:2]
    center = (bound_image.shape[1] // 2, bound_image.shape[0] // 2)
    
    cosofRotationMatrix = numpy.cos(angle*numpy.pi/180)
    sinofRotationMatrix = numpy.sin(angle*numpy.pi/180)
    
    p1 = type("Point", (), {})
    p2 = type("Point", (), {})

    p1.x = height*sinofRotationMatrix + width*cosofRotationMatrix
    p1.y = width*sinofRotationMatrix - height*cosofRotationMatrix
    p1.dx = -cosofRotationMatrix
    p1.dy = -sinofRotationMatrix
    p1.k1 = numpy.abs((p1.x*p1.dy-p1.y*p1.dx)/(width*p1.dy-height*p1.dx))
    p1.k2 = numpy.abs((p1.x*p1.dx+p1.y*p1.dy)/(-width*p1.dx+height*p1.dy))

    p2.x = width*cosofRotationMatrix - height*sinofRotationMatrix
    p2.y = width*sinofRotationMatrix + height*cosofRotationMatrix
    p2.dx = sinofRotationMatrix
    p2.dy = -cosofRotationMatrix
    p2.k1 = numpy.abs((p2.x*p2.dy-p2.y*p2.dx)/(width*p2.dy-height*p2.dx))
    p2.k2 = numpy.abs((p2.x*p2.dx+p2.y*p2.dy)/(-width*p2.dx+height*p2.dy))
    k = min([p1.k1, p1.k2, p2.k1, p2.k2])

    rectangleImageHeight = int(k*height)
    rectangleImageWidth = int(k*width)
    return bound_image[center[1]-(rectangleImageHeight//2):center[1]+(rectangleImageHeight//2), center[0]-(rectangleImageWidth//2):center[0]+(rectangleImageWidth//2)]

if __name__ == "__main__":
    image = cv2.imread(r"C:\Users\MLLD1740\Desktop\image-search\Feature-Detection-and-Matching-master\Figures/Source.png")
    # image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    image = cv2.resize(image, tuple(map(lambda value: int(value / 2), image.shape[1::-1])))

    height, width = image.shape[:2]
    center = (width // 2, height // 2)

    for angle in range(-180, 180+1, 5):
        # Bounding Rectangle
        bound_image = get_bound_image(image, angle)
        BoundingImageHeight, BoundingImageWidth = bound_image.shape[:2]

        # Internal Square Construction
        square_image = get_inner_square_image(bound_image, image, angle)
        squareImageHeight, squareImageWidth = square_image.shape[:2]

        # Internal Rectangle Construction
        a, b = BoundingImageWidth/2+width/2, BoundingImageHeight/2+height/2

        a_hat = (a-BoundingImageWidth/2)*numpy.cos(angle*numpy.pi/180) + (b-BoundingImageHeight/2)*numpy.sin(angle*numpy.pi/180) + BoundingImageWidth/2
        b_hat = (b-BoundingImageHeight/2)*numpy.cos(angle*numpy.pi/180) - (a-BoundingImageWidth/2)*numpy.sin(angle*numpy.pi/180) + BoundingImageHeight/2

        a, b = tuple(map(lambda x: int(x), (a, b)))
        a_hat, b_hat = tuple(map(lambda x: int(x), (a_hat, b_hat)))

        intersections = []
        intersections.extend(list(map(lambda x : (x, (x - a_hat) * numpy.tan(-angle*numpy.pi/180) + b_hat), [a, a-width])))
        intersections.extend(list(map(lambda y : ((y - b_hat) / numpy.tan(-angle*numpy.pi/180) + a_hat, y), [b, b-height])))
        intersections.extend(list(map(lambda x : (x, (x - a_hat) * numpy.tan((-angle+90)*numpy.pi/180) + b_hat), [a, a-width])))
        intersections.extend(list(map(lambda y : ((y - b_hat) / numpy.tan((-angle+90)*numpy.pi/180) + a_hat, y), [b, b-height])))
        intersections.extend([(BoundingImageWidth-x-1, BoundingImageHeight-y-1) for x, y in intersections])                             # Mirror over center point
        intersections = [(x, y) for x, y in intersections if numpy.isfinite(x) and numpy.isfinite(y)]                                   # Filter out non finite values
        intersections_excluded = intersections.copy()
        intersections = [(x, y) for x, y in intersections if a-width-1 <= x and x <= a and b-height-1 <= y and y <= b]                  # Filter out out of bound values
        intersections_excluded = set([(x,y) for x, y in intersections_excluded if (x, y) not in intersections])
        intersections = list(map(lambda point: (int(point[0]), int(point[1])), intersections))                                          # Cast to integer
        intersections = set(intersections)                                                                                              # Remove duplicates

        rectangle_image = get_inner_rectangle_image(bound_image, image, angle)
        rectangleImageHeight, rectangleImageWidth = rectangle_image.shape[:2]
        
        window = 'Rectangle' if rectangleImageHeight*rectangleImageWidth > squareImageHeight*squareImageWidth else "Square"
        print(f"Angle: {angle}")
        print(f"...\t Image")
        print(f"...\t\t Original  height: {height:5d} x width: {width:5d}, area: {height*width:5d}")
        print(f"...\t\t Bounding  height: {BoundingImageHeight:5d} x width: {BoundingImageWidth:5d}, area: {BoundingImageHeight*BoundingImageWidth:5d}")
        print(f"...\t\t Square    height: {squareImageHeight:5d} x width: {squareImageWidth:5d}, area: {squareImageHeight*squareImageWidth:5d}")
        print(f"...\t\t Rectangle height: {rectangleImageHeight:5d} x width: {rectangleImageWidth:5d}, area: {rectangleImageHeight*rectangleImageWidth:5d}")
        print(f"...\t\t Window `{window}`")
        print(f"...\t Anchors")
        print(f"...\t\t (a:     {a:6.2f}, b:     {b:6.2f})")
        print(f"...\t\t (a_hat: {a_hat:6.2f}, b_hat: {b_hat:6.2f})")
        print(f"...\t Center")
        print(f"...\t\t ({BoundingImageWidth/2:6.2f},{BoundingImageHeight/2:6.2f})")
        print(f"...\t Intersections {a-width} <= {a} and {b-height} <= {b}")
        print(f"...\t\t [{len(intersections)}]{intersections}")
        print(f"...\t Excluded")
        print(f"...\t\t [{len(intersections_excluded)}]{intersections_excluded}")

        text = type("text", (), {})()
        text.content = f"Rotation: {angle}"
        text.font = cv2.FONT_HERSHEY_SIMPLEX
        text.fontScale = 1.0
        text.thickness = 2
        text.color=(255, 255, 255)
        text.lineType=cv2.LINE_AA
        text.bottomLeftOrigin = False
        text.size, text.baseline = cv2.getTextSize(text=text.content, fontFace=text.font, fontScale=text.fontScale, thickness=text.thickness)
        text.size = 0, text.size[1]

        output = bound_image.copy()
        cv2.putText(output, text=text.content, org=text.size, fontFace=text.font, fontScale=text.fontScale, color=text.color, thickness=text.thickness, lineType=text.lineType, bottomLeftOrigin=text.bottomLeftOrigin)
        cv2.rectangle(output, (int(BoundingImageWidth/2-width/2), int(BoundingImageHeight/2-height/2)), (int(BoundingImageWidth/2+width/2), int(BoundingImageHeight/2+height/2)), (255, 255, 255), 2)
        cv2.rectangle(output, (int(BoundingImageWidth/2-squareImageWidth/2), int(BoundingImageHeight/2-squareImageHeight/2)), (int(BoundingImageWidth/2+squareImageWidth/2), int(BoundingImageHeight/2+squareImageHeight/2)), (255, 255, 255), 2)
        cv2.rectangle(output, (int(BoundingImageWidth/2-rectangleImageWidth/2), int(BoundingImageHeight/2-rectangleImageHeight/2)), (int(BoundingImageWidth/2+rectangleImageWidth/2), int(BoundingImageHeight/2+rectangleImageHeight/2)), (255, 255, 0), 2)

        cv2.circle(output, (a,b), radius=10, color=(0, 0, 255), thickness=2)
        # cv2.circle(output, (a-width,b), radius=10, color=(0, 255, 0), thickness=2)
        # cv2.circle(output, (a-width,b-height), radius=10, color=(255, 0, 0), thickness=2)
        # cv2.circle(output, (a,b-height), radius=10, color=(0, 255, 255), thickness=2)
        cv2.circle(output, (a_hat,b_hat), radius=10, color=(0, 0, 255), thickness=2)
        cv2.circle(output, (BoundingImageWidth//2,BoundingImageHeight//2), radius=10, color=(0, 0, 0), thickness=2)
        [cv2.circle(output, point, radius=10, color=(255, 255, 0), thickness=2) for point in intersections]
        # for point in intersections:
        #     cv2.line(output, (a_hat, b_hat), point, (255, 0, 0), 5)
        cv2.imshow("Bounded", bound_image)
        cv2.imshow("Square", square_image)
        cv2.imshow("Rectangle", rectangle_image)
        cv2.imshow("Output", output)
        key = cv2.waitKey() & 0xFF
        if key in [27, ord('q'), ord('Q')]:
            break