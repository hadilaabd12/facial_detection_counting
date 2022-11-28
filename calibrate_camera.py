import cv2
import numpy as np

def calibrate_single(imgNums, CheckerboardSize, Nx_cor, Ny_cor):

    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, CheckerboardSize, 1e-6)  # (3,27,1e-6)
    flags = cv2.CALIB_CB_ADAPTIVE_THRESH + cv2.CALIB_CB_FAST_CHECK + cv2.CALIB_CB_NORMALIZE_IMAGE  # 11
    flags_fisheye = cv2.fisheye.CALIB_RECOMPUTE_EXTRINSIC + cv2.fisheye.CALIB_CHECK_COND + cv2.fisheye.CALIB_FIX_SKEW  # 14
 

    objp = np.zeros((1, Nx_cor * Ny_cor, 3), np.float32)
    objp[0, :, :2] = np.mgrid[0:Nx_cor, 0:Ny_cor].T.reshape(-1, 2)
 
    objpoints = []
    imgpoints = []
 
    count = 0
 
    while (True):
        ret, frame = cap.read()
        cv2.imshow('frame', frame)

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        ok, corners = cv2.findChessboardCorners(gray, (Nx_cor, Ny_cor), flags)
        if count >= imgNums:
            break
        if ok:
            objpoints.append(objp)
            cv2.cornerSubPix(gray, corners, (5, 5), (-1, -1), criteria)
            imgpoints.append(corners)

            cv2.drawChessboardCorners(frame, (Nx_cor, Ny_cor), corners, ok)
            count += 1
            print('Find the total number of board corners: ' + str(count))
        
        cv2.waitKey(1)

    global mtx, dist
 
    ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(
        objpoints, imgpoints, gray.shape[:2][::-1], None, criteria
    )

    print('mtx=np.array( ' + str(mtx.tolist()) + " )")
    print('dist=np.array( ' + str(dist.tolist()) + " )")
 
    K = np.zeros((3, 3))
    D = np.zeros((4, 1))
    RR = [np.zeros((1, 1, 3), dtype=np.float64) for i in range(len(objpoints))]
    TT = [np.zeros((1, 1, 3), dtype=np.float64) for i in range(len(objpoints))]
    rms, _, _, _, _ = cv2.fisheye.calibrate(
        objpoints, imgpoints, gray.shape[:2][::-1], K, D, RR, TT, flags_fisheye, criteria
    )

    print("K=np.array( " + str(K.tolist()) + " )")
    print("D=np.array( " + str(D.tolist()) + " )")

    mean_error = 0
    for i in range(len(objpoints)):
        imgpoints2, _ = cv2.projectPoints(objpoints[i], rvecs[i], tvecs[i], mtx, dist)
        error = cv2.norm(imgpoints[i], imgpoints2, cv2.NORM_L2) / len(imgpoints2)
        mean_error += error
    print("Calculate back projection total error: ", mean_error / len(objpoints))
 
    cv2.destroyAllWindows()
    return mtx, dist, K, D

if __name__ == "__main__":
    cap = cv2.VideoCapture(0)

    mtx, dist, K, D = calibrate_single(30, 27, 9, 6)
    # mtx=np.array( [[525.1974085051108, 0.0, 322.46321668550206], [0.0, 470.6897728780676, 207.1415778240149], [0.0, 0.0, 1.0]] )
    # dist=np.array( [[-0.5440259736780028], [0.4582542025510915], [-0.004460196250793969], [-0.010744165783903798], [-0.31459559977372276]] )
    # K=np.array( [[508.94954778109036, 0.0, 308.80041433072194], [0.0, 453.8659706150624, 201.00963020768984], [0.0, 0.0, 1.0]] )
    # D=np.array( [[-0.1710816455825023], [-0.046660635179406704], [0.3972574493629046], [-0.3102470529709773]] )

    width, height = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    p = cv2.fisheye.estimateNewCameraMatrixForUndistortRectify(K, D, (width, height), None)
    mapx2, mapy2 = cv2.fisheye.initUndistortRectifyMap(K, D, None, p, (width, height), cv2.CV_32F)
    while (True):
        ret, frame = cap.read()
        cv2.imshow('frame', frame)
        frame_rectified = cv2.remap(frame, mapx2, mapy2,
            interpolation=cv2.INTER_LINEAR,
            borderMode=cv2.BORDER_CONSTANT
        )
        cv2.imshow('frame_rectified', frame_rectified)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()