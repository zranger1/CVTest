# fix for slow camera initialization on Windows. Doesn't affect Linux or Mac
# but has to go before "import cv2" (I know linters complain about this, but it's necessary)
import os

os.environ["OPENCV_VIDEOIO_MSMF_ENABLE_HW_TRANSFORMS"] = "0"

import time
import cv2
import numpy as np
from pixelblaze import Pixelblaze


class Automap:
    def __init__(self):
        self.ip = "192.168.1.18"  # IP address of Pixelblaze
        self.outFileName = "map.json"  # name of output map file

        # set to True to see extra debug windows and output
        self.debug = True

        self.pb = None
        self.pixelCount = 0
        self.pixelblaze_wait = 0.01
        self.thresholdPct = 0.60
        self.thresholdAdjustDelta = 0.05
        self.maxRetries = 4
        self.framesToSkip = 5
        self.runFlag = True

        # Camera parameters - note, though most cameras can do higher resolutions,
        # I don't recommend using them.  OpenCV slows down a lot as resolution increases.
        self.xSize = 640
        self.ySize = 480
        self.fps = 30

        # initialize camera
        # Note that this can take a *very* long time for some cameras on Windows.
        self.cap = cv2.VideoCapture(0)
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, self.xSize)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.ySize)
        self.cap.set(cv2.CAP_PROP_FPS, self.fps)

        # This might work on some cameras to reduce the number of frames in the ring buffer
        # if it works on your camera, you can reduce the number of frames to skip and go
        # faster!
        # self.cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)

        # we aren't guaranteed the resolution we asked for, so we'll
        # get the current actual resolution of the camera and create empty images to store the
        # camera images and the data we'll generate
        self.xSize = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        self.ySize = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

        self.debug_print("Image size is %d x %d" % (self.xSize, self.ySize))

        # allocate a bunch of image-sized frames for intermediate processing
        self.inputFrame = np.zeros((self.ySize, self.xSize, 3), np.uint8)
        self.workFrame = np.zeros((self.ySize, self.xSize, 3), np.uint8)
        self.background = np.zeros((self.ySize, self.xSize), np.uint8)
        self.grayscale = np.zeros((self.ySize, self.xSize), np.uint8)
        self.threshold = np.zeros((self.ySize, self.xSize), np.uint8)
        self.diff = np.zeros((self.ySize, self.xSize), np.uint8)

    # write a function to print a variable length argument list if self.debug is True
    def debug_print(self, *args):
        if self.debug:
            print(*args)

    @staticmethod
    def is_cuda_available():
        try:
            # Try to import the CUDA module from OpenCV
            _ = cv2.cuda.getCudaEnabledDeviceCount()
            return True
        except Exception:
            return False

    def estimate_threshold(self):
        [minVal, maxVal, minLoc, maxLoc] = cv2.minMaxLoc(self.grayscale)
        r = maxVal - minVal
        thresh = int(maxVal - (r * self.thresholdPct))
        return thresh

    def start_pixelblaze(self, pixelblaze_ip):
        self.pb = Pixelblaze(pixelblaze_ip)
        self.pixelCount = self.pb.getPixelCount()
        self.debug_print("Pixel count is", self.pixelCount)

        # turn off all pixels
        self.all_pixels_off()

    def all_pixels_off(self):
        self.pb.setActiveVariables({"pixel": -1})
        self.wait_for_pixelblaze_pixel(-1)

    def light_pixel(self, pixel: int):
        self.pb.setActiveVariables({"pixel": pixel})
        self.wait_for_pixelblaze_pixel(pixel)

    def wait_for_pixelblaze_pixel(self, pixel):
        time.sleep(self.pixelblaze_wait)

    def get_frame(self, destination, sigma=2.25):
        """
        Get a frame from the camera and convert it to gaussian blurred grayscale
        :param destination:
        :param sigma: sigma value for gaussian blur. higher == more blur
        """
        # skip a few frames to clear the ring buffer to get the most recent image.
        # (Otherwise we wind up with the oldest image in the ring buffer, which can
        # be old enough to cause us to miss lit LEDs.)  This also gives the camera a
        # chance to adjust to new lighting conditions.
        for _ in range(self.framesToSkip):
            if not self.cap.grab():
                break

        # read the next available frame
        self.cap.read(self.inputFrame)

        # Convert to grayscale by taking the red channel only
        # This actually produces a better result for our purpose than using perceptual luma
        _, _, self.workFrame = cv2.split(self.inputFrame)
        # gaussian blur the background frame to reduce the effect of video noise
        cv2.GaussianBlur(self.workFrame, (5, 5), sigma, dst=destination)

    def get_background_frame(self):
        """
        Get a grayscale frame with no lit LEDs for background subtraction.
        """
        self.all_pixels_off()
        self.get_frame(self.background)

    def get_lit_led_frame(self, pixel, lightPixel: bool = True):
        """
        Light the specified LED, get a frame from the camera,
        and generate all the necessary working images.
        :return:
        """

        self.light_pixel(pixel)
        self.get_frame(self.grayscale)

        # Attenuate the background by subtracting the captured background frame.
        # This step cleans up ambient lighting and gets rid of quite a lot of other
        # noise, as well as making it possible to ignore other permanently lit LEDs, like
        # the power indicator on the Pixelblaze itself.
        cv2.absdiff(self.background, self.grayscale, dst=self.diff)

    def circle_led_center(self, cx, cy):
        cv2.circle(self.inputFrame, (cx, cy), 5, (0, 200, 100), 1)

    def putText(self, text, x, y, fontScale=1.0, color=(252, 74, 0), thickness=2):
        """
        Draw text to the camera image buffer
        :param text:
        :param x:
        :param y:
        :param fontScale:
        :param color:
        :param thickness:
        """
        cv2.putText(self.inputFrame, text, (x, y), cv2.FONT_HERSHEY_SIMPLEX,
                    fontScale=fontScale, color=color, thickness=thickness)

    def main(self):
        # if CUDA is available, we'll use it automatically.  Check, and display a message
        # for the user (so if they're expecting CUDA, they can be sure their
        # setup is correct.)
        # print(self.is_cuda_available())

        # connect to the Pixelblaze
        self.start_pixelblaze(self.ip)

        cx = -1
        cy = -1
        pixel = 0

        # We use these values to filter contours based on area
        area = 0
        min_area = 50
        max_area = 3000

        while self.runFlag:
            # capture a frame to use for background subtraction
            # we do this for every new pixel to allow for changes in lighting
            self.get_background_frame()

            # capture a frame with (hopefully) a lit LED and process it
            self.get_lit_led_frame(pixel, True)

            # Threshold the image to isolate the LED
            thresh = self.estimate_threshold()
            cv2.threshold(self.diff, thresh, 255,
                          cv2.THRESH_BINARY, dst=self.threshold)

            found = False

            # Find contours in the thresholded image. If thresholding has worked properly, there will
            # be one big blob of light in the image, and that's the LED we're looking for.
            contours, _ = cv2.findContours(self.threshold, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

            # Use the largest contour
            if len(contours) != 0:
                largest_contour = max(contours, key=cv2.contourArea)
                area = cv2.contourArea(largest_contour)

                if area < min_area:
                    self.debug_print("Contour too small ", area)
                elif area > max_area:
                    self.debug_print("Contour too large ", area)
                else:
                    if min_area < area < max_area:
                        self.debug_print("Pixel:", pixel, " Area:", area)
                        M = cv2.moments(largest_contour)
                        if M["m00"] != 0:
                            cx = int(M["m10"] / M["m00"])
                            cy = int(M["m01"] / M["m00"])

                            found = True

            # as we scan for LED centers, mark them on the input camera frame
            self.circle_led_center(cx, cy)
            self.putText("Pixel %d of %d" % (pixel, self.pixelCount), 420, 30, fontScale=0.5, thickness=1)
            self.putText("Threshold: %f" % self.thresholdPct, 420, 60, fontScale=0.5, thickness=1)
            self.putText("Min Area: %d" % min_area, 420, 90, fontScale=0.5, thickness=1)
            self.putText("Con Area: %d" % area, 420, 120, fontScale=0.5, thickness=1)
            cv2.imshow('Camera', self.inputFrame)

            # debug: display working images in their windows
            if self.debug:
                cv2.imshow("Threshold", self.threshold)
                cv2.imshow("Gray", self.grayscale)
                cv2.imshow("Diff", self.diff)

            # if we don't find anything that looks like an LED, we'll try again
            if not found:
                self.debug_print("No LED found")

            k = cv2.waitKey(0) & 0xFF
            if k == ord(' '):
                self.runFlag = False
            elif k == ord('+'):
                pixel = (pixel + 1) % self.pixelCount
            elif k == ord('-'):
                pixel -= 1
                if pixel < 0:
                    pixel = 255
            elif k == ord('q'):
                self.thresholdPct += self.thresholdAdjustDelta
            elif k == ord('e'):
                self.thresholdPct -= self.thresholdAdjustDelta
            elif k == ord('a'):
                min_area += 10
            elif k == ord('d'):
                min_area -= 10

        # Turn off all LEDs on the Pixelblaze before we exit
        self.all_pixels_off()

        # When everything is done clean up the display and release the capture
        # device
        cv2.destroyAllWindows()
        self.cap.release()


if __name__ == '__main__':
    print("Automap CV Research Version v0.1.0")
    print("Copyright (c) 2024 by ZRanger1. All rights reserved.")
    print("Note: Camera initialization may take up to a minute. Please be patient.")
    print("The camera calibration window will appear when initialization is complete.")
    am = Automap()
    am.main()
