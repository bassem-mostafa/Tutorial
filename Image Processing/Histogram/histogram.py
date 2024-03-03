import cv2
import numpy
from pathlib import Path
from scipy.signal import find_peaks

if __name__ == "__main__":
    # Histogram configuration
    histogram = type("histogram", (), {})                               # create an object for histogram related fields
    histogram.type = "line"                                             # histogram plotting type value from {"line", "bar", "all"}
    histogram.height = 240                                              # histogram canvas height
    histogram.width = 320                                               # histogram canvas width
    histogram.channels = 3                                              # histogram canvas channels
    histogram.bin = type("histogram.bin", (), {})                       # create an object for histogram bin related fields
    histogram.bin.length = 51                                           # histogram number of bins
    histogram.bin.range = (0, 255)                                      # histogram bin values range
    histogram.bin.width = int(histogram.width / histogram.bin.length)   # histogram bin width

    # Histogram compute function
    def _compute(image):
        if len(image.shape) == 2:
            # append 3rd value of shape in case of gray images
            image.shape = *image.shape, 1
        image_histogram = cv2.calcHist(
                                      images=[image],
                                      channels = list(range(image.shape[2])),
                                      mask = None,
                                      histSize = [histogram.bin.length]*image.shape[2],
                                      ranges = list(histogram.bin.range)*image.shape[2],
                                      accumulate = False,
                                      )
        # So far image_histogram represents whole image histogram; example: image_histogram[B=0, G=0, R=0] = number of pixels that have B=0, G=0, R=0
        # Extracting each channel histogram
        if image_histogram.shape[-1] == 1:
            image_histogram.shape = *image_histogram.shape[:-1], # remove last dimension for 1-channel histograms
        channel_histogram = []
        for axis in range(len(image_histogram.shape)):
            axis = tuple(ax for ax in range(len(image_histogram.shape)) if ax != axis)
            channel_histogram.append(numpy.sum(image_histogram, axis=axis))
        # a tuple of histograms for n-channels, or a histogram for 1-channel
        channel_histogram = tuple(channel_histogram) if len(channel_histogram) > 1 else channel_histogram[0]
        return channel_histogram
    histogram.compute = _compute

    # Histogram plot function
    def _plot(bin_values, color):
        bin_values /= numpy.max(bin_values)         # normalize bin values w.r.t. maximum
        bin_values *= histogram.height              # scale to histogram height
        bin_values = histogram.height - bin_values  # invert and shift w.r.t. histogram height so as to plot bottom up instead of top down
        # bin_values = ((bin_values - 0) / (histogram.height - 0)) * (histogram.height - 20) + 10
        bin_values = bin_values.astype(numpy.int32) # cast to integer
        for index, bin_x in enumerate(numpy.arange(0, histogram.bin.length)*histogram.bin.width):
            if histogram.type not in ["line", "bar", "all"]:
                print(f"histogram type `{histogram.type}` not supported")
            if histogram.type in ["bar", "all"]:
                cv2.rectangle(
                    img = histogram.canvas,
                    pt1 = (bin_x, histogram.height),
                    pt2 = (bin_x + histogram.bin.width, bin_values[index]),
                    color = color,
                    thickness = 2,
                    )
            if histogram.type in ["line", "all"]:
                cv2.line(
                    img = histogram.canvas,
                    pt1 = (bin_x, bin_values[index-1] if index > 0 else 0),
                    pt2 = (bin_x + histogram.bin.width, bin_values[index]),
                    color = color,
                    thickness = 2,
                    )
    histogram.plot = _plot

    # Histogram threshold function
    def _threshold(bin_values):
        # compute histogram thresholds
        peaks = find_peaks(bin_values)[0]                                                   # find bin values peaks
        # add first bin as peak if not detected as peak if it has a non-zero value
        if 0 not in peaks and bin_values[0] > 0:
            peaks = numpy.append(peaks, 0)
        # add last bin if not detected as a peak, or only one peak detected
        if histogram.bin.length-1 not in peaks and bin_values[histogram.bin.length-1] > 0:
            peaks = numpy.append(peaks, histogram.bin.length-1)
        threshold_max = numpy.argsort(bin_values[peaks])[-1]                                # find first max bin peak index
        threshold_min = numpy.argsort(bin_values[peaks])[-2]                                # find second max bin peak index
        threshold_max = peaks[threshold_max]                                                # re-map to histogram bin index
        threshold_min = peaks[threshold_min]                                                # re-map to histogram bin index
        if threshold_min > threshold_max:
            # swap if max peak bin lesser than second max peak bin
            threshold_min, threshold_max = threshold_max, threshold_min

        # delta = numpy.abs(threshold_max - threshold_min)                                    # compute delta bins
        # threshold_max = (threshold_max + histogram.bin.length) // 2                         # compute threshold w.r.t. peaks
        # if threshold_min + delta < histogram.bin.length:                                    # compute threshold w.r.t. peaks
        #     # shift minimum if would still in range
        #     threshold_min += delta

        threshold_max = (threshold_max * histogram.bin.range[1]) // histogram.bin.length    # re-scale to bin value range
        threshold_min = (threshold_min * histogram.bin.range[1]) // histogram.bin.length    # re-scale to bin value range
        return tuple([threshold_min, threshold_max])
    histogram.threshold = _threshold

    # Segmentation configuration
    segment = type("segment", (), {})
    segment.kmean = type("segment.kmean", (), {})
    segment.kmean.clusters = 10
    segment.kmean.iterations = 100
    segment.seed = type("segment.seed", (), {})

    # segment using k-means function
    def _kmean_apply(image):
        _compactness, label, center = cv2.kmeans(
                                     data = image.reshape((-1, 1)).astype(numpy.float32),
                                     K = segment.kmean.clusters,
                                     bestLabels = None,
                                     # criteria = ( type, max_iter, epsilon )
                                     # reference: `https://docs.opencv.org/3.4/d1/d5c/tutorial_py_kmeans_opencv.html`
                                     criteria = (
                                                cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER,
                                                segment.kmean.iterations,
                                                1.0,
                                                ),
                                     attempts = 10,
                                     flags = cv2.KMEANS_RANDOM_CENTERS,
                                     )
        center = numpy.uint8(center)                    # convert to numpy array with type uint8
        center[center != numpy.max(center)] = 0         # set zero to all non maximum value
        image = center[label].reshape((image.shape))    # evaluate each label/pixel to its center value
        return image
    segment.kmean.apply = _kmean_apply

    # segment using seed function
    def _seed_apply(image):
        # if len(image.shape) == 2:
        #     # append 3rd value of shape in case of gray images
        #     image.shape = *image.shape, 1
        # cv2.findContours(image, mode, method[, contours[, hierarchy[, offset]]]) → image, contours, hierarchy
        segment_image = image.copy()
        # TODO select suitable seed points
        seed_points = list((x, y) for x in range(image.shape[1]//4) for y in range(image.shape[0]//4))
        seed_points.extend((x, image.shape[0]-y-1) for x in range(image.shape[1]//4) for y in range(image.shape[0]//4))
        seed_points.extend((image.shape[1]-x-1, y) for x in range(image.shape[1]//4) for y in range(image.shape[0]//4))
        seed_points.extend((image.shape[1]-x-1, image.shape[0]-y-1) for x in range(image.shape[1]//4) for y in range(image.shape[0]//4))
        for _segment_seed in seed_points:
            _, segment_image, _, _segment_rectangle = cv2.floodFill(
                                                                  image = segment_image,
                                                                  mask = None,
                                                                  seedPoint = _segment_seed,
                                                                  newVal = (50, 50, 50),
                                                                  loDiff = (50, 50, 50),
                                                                  upDiff = (50, 50, 50),
                                                                  flags = cv2.FLOODFILL_FIXED_RANGE,
                                                                  )
            # cv2.rectangle(segment_image, _segment_rectangle, color = (255, 255, 0), thickness = 1)
            # cv2.circle(segment_image, _segment_seed, radius = 3, color = (255, 0, 0), thickness = -1)
        return segment_image
    segment.seed.apply = _seed_apply

    # traverse on `png` files
    items = Path(__file__).parent.glob("data_rotated/PLANE/image_*.png")
    for item in items:
        # image read and prepare
        image = type("image", (), {})                                                                    # create an object for image related fields
        image.path = item                                                                                # save image path
        image.raw = cv2.imread(str(image.path))                                                          # read image content
        image.gray = cv2.cvtColor(image.raw, cv2.COLOR_BGR2GRAY)                                         # convert to gray
        image.equalized = cv2.equalizeHist(image.gray)                                                   # convert to histogram equalized

        # image segmentation
        image.raw_segment = segment.kmean.apply(image.raw)
        image.gray_segment = segment.kmean.apply(image.gray)
        image.equalized_segment = segment.kmean.apply(image.equalized)

        image.raw_segment = segment.seed.apply(image.raw_segment)
        image.gray_segment = segment.seed.apply(image.gray_segment)
        image.equalized_segment = segment.seed.apply(image.equalized_segment)

        # histogram computation
        histogram.raw = histogram.compute(image.raw)
        histogram.gray = histogram.compute(image.gray)
        histogram.equalized = histogram.compute(image.equalized)
        histogram.raw_segment = histogram.compute(image.raw_segment)
        histogram.gray_segment = histogram.compute(image.gray_segment)
        histogram.equalized_segment = histogram.compute(image.equalized_segment)

        # compute histogram thresholds
        histogram.raw_threshold = tuple(histogram.threshold(histogram.raw[i]) for i in range(len(histogram.raw)))   # computes min, & max threshold for each channel
        histogram.gray_threshold = histogram.threshold(histogram.gray)                                              # computes min, & max threshold
        histogram.equalized_threshold = histogram.threshold(histogram.equalized)                                    # computes min, & max threshold

        # apply histogram thresholds
        # image.raw_threshold = tuple(cv2.threshold(
        #                                          image.raw_segment[:, :, i],
        #                                          histogram.raw_threshold[i][0],
        #                                          histogram.raw_threshold[i][1],
        #                                          cv2.THRESH_TOZERO,
        #                                          )[1] for i in range(len(histogram.raw))
        #                            )
        # image.gray_threshold = cv2.threshold(
        #                                     image.gray_segment,
        #                                     histogram.gray_threshold[0],
        #                                     histogram.gray_threshold[1],
        #                                     cv2.THRESH_TOZERO,
        #                                     )[1]
        # image.equalized_threshold = cv2.threshold(
        #                                     image.equalized_segment,
        #                                     histogram.equalized_threshold[0],
        #                                     histogram.equalized_threshold[1],
        #                                     cv2.THRESH_TOZERO,
        #                                     )[1]
        image.raw_threshold = cv2.split(image.raw_segment)
        image.gray_threshold = image.gray_segment
        image.equalized_threshold = image.equalized_segment

        # post-processing
        # de-noising using median
        for _ in range(1):
            image.raw_threshold = tuple(cv2.medianBlur(image.raw_threshold[i], 3) for i in range(len(histogram.raw)))
            image.gray_threshold = cv2.medianBlur(image.gray_threshold, 3)
            image.equalized_threshold = cv2.medianBlur(image.equalized_threshold, 3)

        # image restoration
        morphology = type("morphology", (), {})
        morphology.dilate = type("morphology_dilate", (), {})
        morphology.dilate.kernel = numpy.ones((3, 3), numpy.uint8)
        morphology.dilate.iterations = 1
        morphology.erode = type("morphology_erode", (), {})
        morphology.erode.kernel = numpy.ones((3, 3), numpy.uint8)
        morphology.erode.iterations = 1
        morphology.open = type("morphology_open", (), {})
        morphology.open.kernel = numpy.ones((3, 3), numpy.uint8)
        morphology.open.iterations = 1
        morphology.close = type("morphology_close", (), {})
        morphology.close.kernel = numpy.ones((3, 3), numpy.uint8)
        morphology.close.iterations = 1

        # image.raw_threshold = tuple(cv2.dilate(image.raw_threshold[i], morphology.dilate.kernel, iterations=morphology.dilate.iterations) for i in range(len(histogram.raw)))
        # image.gray_threshold = cv2.dilate(image.gray_threshold, morphology.dilate.kernel, iterations=morphology.dilate.iterations)
        # image.equalized_threshold = cv2.dilate(image.equalized_threshold, morphology.dilate.kernel, iterations=morphology.dilate.iterations)
        #
        # image.raw_threshold = tuple(cv2.erode(image.raw_threshold[i], morphology.erode.kernel, iterations=morphology.erode.iterations) for i in range(len(histogram.raw)))
        # image.gray_threshold = cv2.erode(image.gray_threshold, morphology.erode.kernel, iterations=morphology.erode.iterations)
        # image.equalized_threshold = cv2.erode(image.equalized_threshold, morphology.erode.kernel, iterations=morphology.erode.iterations)

        # for _ in range(1):
        #     image.raw_threshold = tuple(cv2.morphologyEx(image.raw_threshold[i], cv2.MORPH_OPEN, morphology.open.kernel, iterations=morphology.open.iterations) for i in range(len(histogram.raw)))
        #     image.gray_threshold = cv2.morphologyEx(image.gray_threshold, cv2.MORPH_OPEN, morphology.open.kernel, iterations=morphology.open.iterations)
        #     image.equalized_threshold = cv2.morphologyEx(image.equalized_threshold, cv2.MORPH_OPEN, morphology.open.kernel, iterations=morphology.open.iterations)
        #
        #     image.raw_threshold = tuple(cv2.medianBlur(image.raw_threshold[i], 3) for i in range(len(histogram.raw)))
        #     image.gray_threshold = cv2.medianBlur(image.gray_threshold, 3)
        #     image.equalized_threshold = cv2.medianBlur(image.equalized_threshold, 3)
        #
        #     image.raw_threshold = tuple(cv2.morphologyEx(image.raw_threshold[i], cv2.MORPH_CLOSE, morphology.close.kernel, iterations=morphology.close.iterations) for i in range(len(histogram.raw)))
        #     image.gray_threshold = cv2.morphologyEx(image.gray_threshold, cv2.MORPH_CLOSE, morphology.close.kernel, iterations=morphology.close.iterations)
        #     image.equalized_threshold = cv2.morphologyEx(image.equalized_threshold, cv2.MORPH_CLOSE, morphology.close.kernel, iterations=morphology.close.iterations)
        #
        #     image.raw_threshold = tuple(cv2.medianBlur(image.raw_threshold[i], 3) for i in range(len(histogram.raw)))
        #     image.gray_threshold = cv2.medianBlur(image.gray_threshold, 3)
        #     image.equalized_threshold = cv2.medianBlur(image.equalized_threshold, 3)

        # histogram equalization
        image.raw_threshold = tuple(cv2.equalizeHist(image.raw_threshold[i]) for i in range(len(histogram.raw)))
        image.gray_threshold = cv2.equalizeHist(image.gray_threshold)
        image.equalized_threshold = cv2.equalizeHist(image.equalized_threshold)

        text = type("text", (), {})()
        text.font = cv2.FONT_HERSHEY_SIMPLEX
        text.fontScale = 0.5
        text.thickness = 2
        text.color=(0, 255, 0)
        text.lineType=cv2.LINE_AA
        text.bottomLeftOrigin = False

        # prepare histogram canvas
        histogram.canvas = numpy.zeros((histogram.height, histogram.width, histogram.channels), dtype=numpy.uint8)

        # plot thresholds
        histogram.raw_threshold = tuple(tuple((histogram.raw_threshold[i][j] * histogram.bin.length * histogram.bin.width) // histogram.bin.range[1] for j in range(2)) for i in range(len(histogram.raw))) # re-scale to histogram canvas range
        histogram.gray_threshold = tuple((histogram.gray_threshold[j] * histogram.bin.length * histogram.bin.width) // histogram.bin.range[1] for j in range(2))                                     # re-scale to histogram canvas range
        histogram.equalized_threshold = tuple((histogram.equalized_threshold[j] * histogram.bin.length * histogram.bin.width) // histogram.bin.range[1] for j in range(2))                                 # re-scale to histogram canvas range
        # for threshold_min, threshold_max in [*histogram.raw_threshold, histogram.gray_threshold, histogram.equalized_threshold]:
        for threshold_min, threshold_max in [histogram.equalized_threshold]:
            cv2.line(
                img = histogram.canvas,
                pt1 = (threshold_min, 0),
                pt2 = (threshold_min, histogram.height),
                color = (0.0, 255, 255),
                thickness = 5,
                )
            cv2.line(
                img = histogram.canvas,
                pt1 = (threshold_max, 0),
                pt2 = (threshold_max, histogram.height),
                color = (255, 0, 255),
                thickness = 5,
                )

        # plot histograms
        histogram.plot(histogram.raw[0], (255, 0, 0))                                                   # plot axis 0: Blue histogram
        histogram.plot(histogram.raw[1], (0, 255, 0))                                                   # plot axis 1: Green histogram
        histogram.plot(histogram.raw[2], (0, 0, 255))                                                   # plot axis 2: Red histogram
        histogram.plot(histogram.gray, (255, 255, 255))                                                 # plot axis 0: Gray histogram
        histogram.plot(histogram.equalized, (125, 125, 125))                                            # plot axis 0: equalized histogram

        # histogram.plot(histogram.raw_segment[0], (255, 0, 0))                                           # plot axis 0: Blue histogram
        # histogram.plot(histogram.raw_segment[1], (0, 255, 0))                                           # plot axis 1: Green histogram
        # histogram.plot(histogram.raw_segment[2], (0, 0, 255))                                           # plot axis 2: Red histogram
        # histogram.plot(histogram.gray_segment, (255, 255, 255))                                         # plot axis 0: Gray histogram
        # histogram.plot(histogram.equalized_segment, (125, 125, 125))                                    # plot axis 0: equalized histogram

        # prepare image canvas
        # image.raw # no need for conversion
        image.gray = cv2.cvtColor(image.gray, cv2.COLOR_GRAY2BGR)
        image.equalized = cv2.cvtColor(image.equalized, cv2.COLOR_GRAY2BGR)
        # image.raw_segment # no need for conversion
        image.gray_segment = cv2.cvtColor(image.gray_segment, cv2.COLOR_GRAY2BGR)
        image.equalized_segment = cv2.cvtColor(image.equalized_segment, cv2.COLOR_GRAY2BGR)
        image.raw_threshold = cv2.merge(image.raw_threshold)
        image.gray_threshold = cv2.cvtColor(image.gray_threshold, cv2.COLOR_GRAY2BGR)
        image.equalized_threshold = cv2.cvtColor(image.equalized_threshold, cv2.COLOR_GRAY2BGR)
        image.canvas = cv2.hconcat([image.raw, image.gray, image.equalized])
        image.canvas = cv2.vconcat([image.canvas, cv2.hconcat([image.raw_segment, image.gray_segment, image.equalized_segment])])
        image.canvas = cv2.vconcat([image.canvas, cv2.hconcat([image.raw_threshold, image.gray_threshold, image.equalized_threshold])])

        # write image information into image canvas
        text.content = f"{image.path.stem}"
        text.size, text.baseline = cv2.getTextSize(text=text.content, fontFace=text.font, fontScale=text.fontScale, thickness=text.thickness)
        text.size = 0, text.size[1]
        image.canvas = cv2.vconcat([numpy.zeros((text.size[1] + text.baseline, image.canvas.shape[1], image.canvas.shape[2]), dtype=numpy.uint8), image.canvas])
        cv2.putText(image.canvas, text=text.content, org=text.size, fontFace=text.font, fontScale=text.fontScale, color=text.color, thickness=text.thickness, lineType=text.lineType, bottomLeftOrigin=text.bottomLeftOrigin)

        # show output figure(s)
        cv2.imshow("image", image.canvas)
        cv2.imshow("histogram", histogram.canvas)
        key = cv2.waitKey() & 0xFF
        if key in [27, ord("Q"), ord("q")]:
            break