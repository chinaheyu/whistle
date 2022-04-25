# coding=utf-8
import cv2
import numpy as np
import mvsdk
import platform
import time


class MonocularCamera(object):
    def __init__(self, config_file=None):
        super(MonocularCamera, self).__init__()
        DevList = mvsdk.CameraEnumerateDevice()
        nDev = len(DevList)
        if nDev < 1:
            print("No camera was found!")
            exit(-1)
        elif nDev == 1:
            DevInfo = DevList[0]
        else:
            for i, DevInfo in enumerate(DevList):
                print("{}: {} {}".format(i, DevInfo.GetFriendlyName(), DevInfo.GetPortType()))
            i = int(input("Select cameras: "))
            DevInfo = DevList[i]
        self.DevInfo = DevInfo
        self.hCamera = 0
        self.cap = None
        self.pFrameBuffer = 0
        self.open(config_file)

    def open(self, config_file=None):
        if self.hCamera > 0:
            return True

        # 打开相机
        hCamera = 0
        try:
            hCamera = mvsdk.CameraInit(self.DevInfo, -1, -1)
        except mvsdk.CameraException as e:
            print("CameraInit Failed({}): {}".format(e.error_code, e.message))
            return False

        # 获取相机特性描述
        cap = mvsdk.CameraGetCapability(hCamera)

        # 判断是黑白相机还是彩色相机
        monoCamera = (cap.sIspCapacity.bMonoSensor != 0)

        # 黑白相机让ISP直接输出MONO数据，而不是扩展成R=G=B的24位灰度
        if monoCamera:
            mvsdk.CameraSetIspOutFormat(hCamera, mvsdk.CAMERA_MEDIA_TYPE_MONO8)
        else:
            mvsdk.CameraSetIspOutFormat(hCamera, mvsdk.CAMERA_MEDIA_TYPE_BGR8)

        if config_file is not None:
            if mvsdk.CameraReadParameterFromFile(hCamera, config_file) == 0:
                print('Read config file success.')
            else:
                print('Cannot read config file.')

        # 计算RGB buffer所需的大小，这里直接按照相机的最大分辨率来分配
        FrameBufferSize = cap.sResolutionRange.iWidthMax * cap.sResolutionRange.iHeightMax * (1 if monoCamera else 3)

        # 分配RGB buffer，用来存放ISP输出的图像
        # 备注：从相机传输到PC端的是RAW数据，在PC端通过软件ISP转为RGB数据（如果是黑白相机就不需要转换格式，但是ISP还有其它处理，所以也需要分配这个buffer）
        pFrameBuffer = mvsdk.CameraAlignMalloc(FrameBufferSize, 16)

        # 让SDK内部取图线程开始工作
        mvsdk.CameraPlay(hCamera)

        self.hCamera = hCamera
        self.pFrameBuffer = pFrameBuffer
        self.cap = cap
        return True

    def close(self):
        if self.hCamera > 0:
            mvsdk.CameraUnInit(self.hCamera)
            self.hCamera = 0

        mvsdk.CameraAlignFree(self.pFrameBuffer)
        self.pFrameBuffer = 0

    def grab(self):
        # 从相机取一帧图片
        hCamera = self.hCamera
        pFrameBuffer = self.pFrameBuffer
        try:
            pRawData, FrameHead = mvsdk.CameraGetImageBuffer(hCamera, 200)
            mvsdk.CameraImageProcess(hCamera, pRawData, pFrameBuffer, FrameHead)
            mvsdk.CameraReleaseImageBuffer(hCamera, pRawData)

            # windows下取到的图像数据是上下颠倒的，以BMP格式存放。转换成opencv则需要上下翻转成正的
            # linux下直接输出正的，不需要上下翻转
            if platform.system() == "Windows":
                mvsdk.CameraFlipFrameBuffer(pFrameBuffer, FrameHead, 1)

            # 此时图片已经存储在pFrameBuffer中，对于彩色相机pFrameBuffer=RGB数据，黑白相机pFrameBuffer=8位灰度数据
            # 把pFrameBuffer转换成opencv的图像格式以进行后续算法处理
            frame_data = (mvsdk.c_ubyte * FrameHead.uBytes).from_address(pFrameBuffer)
            frame = np.frombuffer(frame_data, dtype=np.uint8)
            frame = frame.reshape((FrameHead.iHeight, FrameHead.iWidth,
                                   1 if FrameHead.uiMediaType == mvsdk.CAMERA_MEDIA_TYPE_MONO8 else 3))
            return frame
        except mvsdk.CameraException as e:
            if e.error_code != mvsdk.CAMERA_STATUS_TIME_OUT:
                print("CameraGetImageBuffer failed({}): {}".format(e.error_code, e.message))
            return None


def main_loop():
    cam = MonocularCamera()
    st = time.time()

    while (cv2.waitKey(1) & 0xFF) != ord('q'):
        frame = cam.grab()
        if frame is not None:
            frame = cv2.resize(frame, (640, 480), interpolation=cv2.INTER_LINEAR)
            et = time.time()
            cv2.putText(frame, f"FPS: {1 / (et - st):.2f}", (0, 24), cv2.FONT_HERSHEY_COMPLEX, 1, (0, 121, 242), 2)
            st = et
            cv2.imshow("{} Press q to end".format(cam.DevInfo.GetFriendlyName()), frame)

    cam.close()


def main():
    try:
        main_loop()
    finally:
        cv2.destroyAllWindows()


if __name__ == '__main__':
    main()
