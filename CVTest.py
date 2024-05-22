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
        self.maxRetries = 6
        self.framesToSkip = 5

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

    def calibrate_camera(self):
        """
        Allow the user to aim the camera by displaying the camera feed and waiting for a keypress.
        :return:
        """
        while True:
            self.cap.read(self.inputFrame)
            self.putText("Aim Camera. Press <space> to start", 20, 30)
            cv2.imshow('Camera', self.inputFrame)

            if cv2.waitKey(1) & 0xFF == ord(' '):
                break

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

        # if retrying, we don't need to change the lighting
        # this will save us from potential camera timing errors
        if lightPixel:
            self.light_pixel(pixel)

        self.get_frame(self.grayscale)

        # Attenuate the background by subtracting the captured background frame.
        # This step cleans up ambient lighting and gets rid of quite a lot of other
        # noise, as well as making it possible to ignore other permanently lit LEDs, like
        # the power indicator on the Pixelblaze itself.
        cv2.absdiff(self.background, self.grayscale, dst=self.diff)

    def circle_led_centers(self, led_centers):
        for (x, y) in led_centers:
            cv2.circle(self.inputFrame, (x, y), 5, (0, 200, 100), 1)

    def mark_display_center(self, x_center, y_center, axisSize=100):
        """
        Mark the center of the display with a crosshair
        :param x_center:
        :param y_center:
        :param axisSize:
        :return:
        """
        cv2.line(self.inputFrame, (x_center - axisSize, y_center), (x_center + axisSize, y_center), (0, 0, 200), )
        cv2.line(self.inputFrame, (x_center, y_center - axisSize), (x_center, y_center + axisSize), (0, 0, 200), 2)

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

    @staticmethod
    def rotate_led_centers(led_centers, x_center, y_center, radians):
        """
        Rotate the LED centers about the center of the display by the specified angle to create
        a new list.
        :param led_centers:
        :param x_center:
        :param y_center:
        :param radians:
        :return: a new, rotated list of LED centers
        """
        new_centers = []
        for led in led_centers:
            # ignore any LEDs that we weren't able to see
            if led[0] < 0 and led[1] < 0:
                continue

            x = led[0] - x_center
            y = led[1] - y_center
            x_new = int(x * np.cos(radians) - y * np.sin(radians) + x_center)
            y_new = int(x * np.sin(radians) + y * np.cos(radians) + y_center)
            new_centers.append([x_new, y_new])

        return new_centers

    def main(self):
        # if CUDA is available, we'll use it automatically.  Check, and display a message
        # for the user (so if they're expecting CUDA, they can be sure their
        # setup is correct.)
        # print(self.is_cuda_available())

        # connect to the Pixelblaze
        self.start_pixelblaze(self.ip)

        # give the user a chance to aim the camera correctly
        self.calibrate_camera()

        led_centers = []
        pixel = 0

        # We use these values to filter contours based on area
        min_area = 200
        max_area = 2000
        retry = 0

        while pixel < self.pixelCount:

            # if we run out of retries, the pixel probably isn't visible
            # we'll skip it and assign it to an easy-to-spot location so the user
            # can edit it manually later.
            if retry >= self.maxRetries:
                self.debug_print("Out of retries - Skipping pixel (coords assigned to [-1,-1])")
                led_centers.append((-1, -1))
                pixel += 1
                retry = 0

            # capture a frame to use for background subtraction
            # we do this for every new pixel to allow for changes in lighting
            if retry == 0:
                self.get_background_frame()

            # capture a frame with (hopefully) a lit LED and process it
            self.get_lit_led_frame(pixel, retry == 0)

            # Threshold the image to isolate the LED
            # TODO - need to build a custom adaptive thresholding function for this.  The functions
            # TODO - available in OpenCV stuff doesn't quite do the right thing for this special case.
            thresh = self.estimate_threshold()
            cv2.threshold(self.diff, thresh, 255,
                          cv2.THRESH_BINARY, dst=self.threshold)

            found = False
            self.debug_print("Pixel %d" % pixel, "Threshold:", thresh, "Retry:", retry)

            # Find contours in the thresholded image. If thresholding has worked properly, there will
            # be one big blob of light in the image, and that's the LED we're looking for.
            contours, _ = cv2.findContours(self.threshold, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

            # Use the largest contour
            if len(contours) != 0:
                largest_contour = max(contours, key=cv2.contourArea)
                area = cv2.contourArea(largest_contour)

                # find aspect ratio of largest contour
                x, y, w, h = cv2.boundingRect(largest_contour)
                x = min(w, h)
                y = max(w, h)
                aspect_ratio = float(x) / float(y)
                self.debug_print("Aspect ratio: ", aspect_ratio)

                if area < min_area:
                    # if the contour is too small, we'll try again with a different threshold
                    self.thresholdPct += self.thresholdAdjustDelta
                    self.debug_print("Contour too small ", area, " Adjusting threshold to ", self.thresholdPct)
                    retry += 1
                elif area > max_area:
                    # if the contour is too big, it might to be an actual problem
                    # with large reflections or ambient light.  Before we give up, we'll modify
                    # the threshold to see if we can get a better result.
                    self.thresholdPct -= self.thresholdAdjustDelta
                    self.debug_print("Contour too large ", area, " Adjusting threshold to ", self.thresholdPct)
                    retry += 1
                else:
                    # find bounding box, center and aspect ratio of largest contour
                    x, y, w, h = cv2.boundingRect(largest_contour)
                    center_x = x + w // 2
                    center_y = y + h // 2
                    x = min(w, h)
                    y = max(w, h)
                    aspect_ratio = float(x) / float(y)

                    # we're looking for a roughly circular region, which means an aspect ratio
                    # of about 1.  If the box is long and skinny, we adjust the threshold and try again.
                    if aspect_ratio < 0.6:
                        self.thresholdPct += self.thresholdAdjustDelta
                        self.debug_print("Aspect ratio too low. Adjusting threshold to ", self.thresholdPct)
                        retry += 1
                    else:
                        led_centers.append((center_x, center_y))
                        self.debug_print("Accepted w/ area:", area)
                        # self.debug_print("Aspect ratio: ", aspect_ratio)
                        found = True
            else:
                # if we don't find any contours, we'll try again with a different threshold
                self.thresholdPct += self.thresholdAdjustDelta
                self.debug_print("No contours found. Adjusting threshold to ", self.thresholdPct)
                retry += 1

            # as we scan for LED centers, mark them on the input camera frame
            self.circle_led_centers(led_centers)
            self.putText("Pixel %d of %d" % (pixel, self.pixelCount), 460, 30, fontScale=0.5, thickness=1)
            cv2.imshow('Camera', self.inputFrame)

            # debug: display working images in their windows
            if self.debug:
                cv2.imshow("Threshold", self.threshold)
                cv2.imshow("Gray", self.grayscale)
                cv2.imshow("Diff", self.diff)

            # if we've located a pixel center, move on to the next one
            if found:
                pixel += 1
                retry = 0

            if cv2.waitKey(1) & 0xFF == ord('q'):
                exit(0)

        # done scanning for LEDs!
        # convert led_centers to a list of lists for JSON output
        led_centers = [list(t) for t in led_centers]

        # find x,y center point of all the identified LEDs in led_centers list
        x_center = 0
        y_center = 0
        count = len(led_centers)
        for led in led_centers:

            # ignore any LEDs that we weren't able to see
            # (these will have coordinates of [-1,-1])
            if led[0] < 0 and led[1] < 0:
                count -= 1
                continue
            x_center += led[0]
            y_center += led[1]

        x_center = x_center // count
        y_center = y_center // count

        # Display the final result and let the user rotate it to a good
        # orientation using the "a" and "d" keys.  Press <space> to save the map.
        theta = 0
        while True:
            self.cap.read(self.inputFrame)

            new_centers = self.rotate_led_centers(led_centers, x_center, y_center, theta)
            self.circle_led_centers(new_centers)
            self.mark_display_center(x_center, y_center)
            self.putText("Rotate with 'a' and 'd' keys", 80, 30)
            self.putText("Press <space> to save map", 100, 60)
            cv2.imshow('Camera', self.inputFrame)
            k = cv2.waitKey(1) & 0xFF
            if k == ord(' '):
                break
            elif k == ord('a'):
                # rotate left
                theta -= 0.0628
            elif k == ord('d'):
                # rotate right
                theta += 0.0628

        # write led_centers to output file in JSON format for use
        # as a Pixelblaze map.
        with open(self.outFileName, 'w') as f:
            f.write(str(new_centers))

        print("Saved map with %d LEDs to %s" % (len(new_centers), self.outFileName))

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
