from hog_utils import *
import matplotlib.image as mpimg
import matplotlib.pyplot as plt
import numpy as np
import pickle
import cv2

#chagne

[X_scaler,svc] = pickle.load(open("model_YCrCb__ALL.p", "rb"))
orient = 9
pix_per_cell = 8
cell_per_block = 2
spatial_size = 16
hist_bins = 32
#spatial_size = dist_pickle["spatial_size"]
#hist_bins = dist_pickle["hist_bins"]


# Define a single function that can extract features using hog sub-sampling and make predictions
def find_cars(img, ystart, ystop, scale, svc, X_scaler, orient, pix_per_cell, cell_per_block, spatial_size, hist_bins,cells_per_step = 2):
    draw_img = np.copy(img)
    #img = img.astype(np.float32) / 255 # I dont change to (0,255) because it is normailzed by scaler
    rectangles = []
    img_tosearch = img[ystart:ystop, :, :]
    ctrans_tosearch = cv2.cvtColor(img_tosearch,cv2.COLOR_BGR2YCrCb)
    if scale != 1:
        imshape = ctrans_tosearch.shape
        ctrans_tosearch = cv2.resize(ctrans_tosearch, (np.int(imshape[1] / scale), np.int(imshape[0] / scale)))

    ch1 = ctrans_tosearch[:, :, 0]
    ch2 = ctrans_tosearch[:, :, 1]
    ch3 = ctrans_tosearch[:, :, 2]

    # Define blocks and steps as above
    nxblocks = (ch1.shape[1] // pix_per_cell) - cell_per_block + 1
    nyblocks = (ch1.shape[0] // pix_per_cell) - cell_per_block + 1
    nfeat_per_block = orient * cell_per_block ** 2

    # 64 was the orginal sampling rate, with 8 cells and 8 pix per cell
    window = 64
    nblocks_per_window = (window // pix_per_cell) - cell_per_block + 1
      # Instead of overlap, define how many cells to step
    nxsteps = (nxblocks - nblocks_per_window) // cells_per_step +1
    nysteps = (nyblocks - nblocks_per_window) // cells_per_step +1

    # Compute individual channel HOG features for the entire image
    hog1 = get_hog_features(ch1, orient, pix_per_cell, cell_per_block, feature_vec=False)
    hog2 = get_hog_features(ch2, orient, pix_per_cell, cell_per_block, feature_vec=False)
    hog3 = get_hog_features(ch3, orient, pix_per_cell, cell_per_block, feature_vec=False)

    for xb in range(nxsteps):
        for yb in range(nysteps):
            ypos = yb * cells_per_step
            xpos = xb * cells_per_step
            # Extract HOG for this patch
            hog_feat1 = hog1[ypos:ypos + nblocks_per_window, xpos:xpos + nblocks_per_window].ravel()
            hog_feat2 = hog2[ypos:ypos + nblocks_per_window, xpos:xpos + nblocks_per_window].ravel()
            hog_feat3 = hog3[ypos:ypos + nblocks_per_window, xpos:xpos + nblocks_per_window].ravel()
            hog_features = np.hstack((hog_feat1, hog_feat2, hog_feat3))

            xleft = xpos * pix_per_cell
            ytop = ypos * pix_per_cell

            # Extract the image patch
            subimg = cv2.resize(ctrans_tosearch[ytop:ytop + window, xleft:xleft + window], (64, 64))

            # Get color features
            spatial_features = bin_spatial(subimg, size=(spatial_size,spatial_size))
            hist_features = color_hist(subimg, nbins=hist_bins)

            # Scale features and make a prediction
            test_features = X_scaler.transform(
                np.hstack((hog_features, spatial_features, hist_features)).reshape(1, -1))
            # test_features = X_scaler.transform(np.hstack((shape_feat, hist_feat)).reshape(1, -1))
            test_prediction = svc.predict(test_features)
            #test_prediction=1
            if test_prediction == 1:
                xbox_left = np.int(xleft * scale)
                ytop_draw = np.int(ytop * scale)
                win_draw = np.int(window * scale)
                rectangles.append([(xbox_left, ytop_draw + ystart),
                              (xbox_left + win_draw, ytop_draw + win_draw + ystart)])
    return rectangles
    #return draw_img

def search_cars(img,areas):
    rectangles=[]
    for area in areas:
        rc = find_cars(img, area[0], area[1], area[2], svc, X_scaler, orient, pix_per_cell, cell_per_block, spatial_size,
                       hist_bins,cells_per_step=area[3])
        if len(rc) > 0:
            if len(rectangles) > 0:
                rectangles = np.vstack((rectangles, rc))
            else:
                rectangles = rc
    return rectangles

def procTestImages():
    img0=[]
    img1=[]
    for i in [1,2,3,4,5,6]:
        img = cv2.imread('test_images/test'+str(i)+'.jpg')
        heat = np.zeros_like(img[:, :, 0]).astype(np.float)
        areas = [[350,625,2.5,2],[350,558,2,2],[400, 540, 1.5,4],[400, 512, 1,2],[400, 464, .5,8]]
        rectangles1 = search_cars(img,[areas[0]])
        rectangles2 = search_cars(img, [areas[1]])
        rectangles3 = search_cars(img, [areas[2]])
        rectangles4 = search_cars(img, [areas[3]])
        rectangles5 = search_cars(img, [areas[4]])
        rectangles = rectangles1 + rectangles2 + rectangles3 + rectangles4 + rectangles5
        if len(rectangles) > 0:
            heat = add_heat(heat, rectangles)
            heat = apply_threshold(heat, 1)
            heatmap = np.clip(heat, 0, 255)
            labels = label(heatmap)
            out_img = img.copy()
            for rec in rectangles1:
                cv2.rectangle(out_img, tuple(rec[0]),
                              tuple(rec[1]), (255, 255, 0), 2)
            for rec in rectangles2:
                cv2.rectangle(out_img, tuple(rec[0]),
                              tuple(rec[1]), (0, 255, 0), 2)
            for rec in rectangles3:
                cv2.rectangle(out_img, tuple(rec[0]),
                              tuple(rec[1]), (0, 0, 255), 2)
            for rec in rectangles4:
                cv2.rectangle(out_img, tuple(rec[0]),
                              tuple(rec[1]), (255, 0, 255), 2)
            for rec in rectangles5:
                cv2.rectangle(out_img, tuple(rec[0]),
                              tuple(rec[1]), (0,255, 255), 2)
            draw_img = draw_labeled_bboxes(img, labels)
            cv2.imwrite("output_images/positives_"+str(i)+".png",out_img)
            cv2.imwrite("output_images/detections_" + str(i) + ".png", draw_img)
        else:
            cv2.imwrite("output_images/positives_" + str(i) + ".png", img)
            cv2.imwrite("output_images/detections_" + str(i) + ".png", img)
#with the next two lines I can get pictures of widnows detections. For video I comment it
#procTestImages()
#exit(0)

cleanImage = False
videoFileName = 'project_video.mp4'
cap = cv2.VideoCapture(videoFileName)
cap.set(cv2.CAP_PROP_POS_FRAMES, 250)
[valid, img] = cap.read()
heat = np.zeros_like(img[:, :, 0]).astype(np.float)
frame = 0
while valid:
    frame+=1
    #areas = [[350,625,2.5,2],[350,558,2,2],[400, 540, 1.5,4],[400, 512, 1,8],[400, 464, .5,8]]
    areas = [[350, 625, 2.5, 2], [350, 558, 2, 2], [400, 540, 1.5, 4], [400, 512, 1, 2], [400, 464, .5, 8]]
    heat = heat / 2
    rectangles1 = search_cars(img,[areas[0]])
    rectangles2 = search_cars(img, [areas[1]])
    rectangles3 = search_cars(img, [areas[2]])
    rectangles4 = search_cars(img, [areas[3]])
    rectangles5 = search_cars(img, [areas[4]])
    rectangles = rectangles1 + rectangles2 +rectangles3 +rectangles4 +rectangles5
    if len(rectangles) > 0:
        heat = add_heat(heat, rectangles)
        heat = apply_threshold(heat, 2)
        heatmap = np.clip(heat, 0, 255)
        labels = label(heatmap)
        out_img = img.copy()

        for rec in rectangles1:
            cv2.rectangle(out_img, tuple(rec[0]),
                          tuple(rec[1]), (255, 255, 0), 2)
        for rec in rectangles2:
            cv2.rectangle(out_img, tuple(rec[0]),
                          tuple(rec[1]), (0, 255, 0), 2)
        for rec in rectangles3:
            cv2.rectangle(out_img, tuple(rec[0]),
                          tuple(rec[1]), (0, 0, 255), 2)
        for rec in rectangles4:
            cv2.rectangle(out_img, tuple(rec[0]),
                          tuple(rec[1]), (255, 0, 255), 2)
        for rec in rectangles5:
            cv2.rectangle(out_img, tuple(rec[0]),
                          tuple(rec[1]), (0,255, 255), 2)
        # replace img by out_img if you want to see positive windows.
        draw_img = draw_labeled_bboxes(img, labels)
        cv2.imshow("ventana", draw_img)
        cv2.imwrite("finalVideo/frame"+str(frame)+".png",draw_img)
        cv2.waitKey(1)
    else:
        cv2.imshow("ventana", img)
        cv2.imwrite("finalVideo/frame" + str(frame) + ".png", draw_img)
        cv2.waitKey(1)
        # fig = plt.figure()
        # plt.subplot(121)
        # plt.imshow(draw_img)
        # plt.title('Car Positions')
        # plt.subplot(122)
        # plt.imshow(heatmap, cmap='hot')
        # plt.title('Heat Map')
        # plt.draw()
        # fig.tight_layout()
    #reduce image processing jumping 3 frames in each iteration
    for i in range(3):
        [valid, img] = cap.read()




        #plt.imshow(out_img)
        #cv2.imshow("ventana",out_img)
        #    if i==1:
        #        cv2.imwrite("mulpiple_sizes_detection.jpg",out_img)
        #cv2.waitKey(1)


pass




