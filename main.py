import numpy as np
import cv2
import random

from threading import Thread
import time
import queue
import threading
import math

import pygame
from pygame.locals import *

from playsound import playsound

cap = cv2.VideoCapture(0)

fgbg = cv2.bgsegm.createBackgroundSubtractorMOG()

lock = threading.Lock()

# Will be overriden below
#positionText = (10, 50)
colorText = (0, 0, 0)
color = (0, 0, 255)

# Current frame
drawing = None

imgH = 100
imgW = 100

tokenCount = 3
maxSpeed = 5

screenWidth = 0

class Game:
    def __init__(self):
        self.points = 0
        self.time = 0

class Token:
    # speed: 1 (slow) to 10 (fast)
    def __init__(self, x, y, speed):
        self.x = x
        self.y = y
        self.speed = speed

class Danger:
    def __init__(self, x):
        self.x = x
        self.y = 0
        self.speed = 3

game = Game()
tokens = []
danger = None

def add_transparent_image(background, foreground, x_offset=None, y_offset=None):
    bg_h, bg_w, bg_channels = background.shape
    fg_h, fg_w, fg_channels = foreground.shape

    assert bg_channels == 3, f'background image should have exactly 3 channels (RGB). found:{bg_channels}'
    assert fg_channels == 4, f'foreground image should have exactly 4 channels (RGBA). found:{fg_channels}'

    # center by default
    if x_offset is None: x_offset = (bg_w - fg_w) // 2
    if y_offset is None: y_offset = (bg_h - fg_h) // 2

    w = min(fg_w, bg_w, fg_w + x_offset, bg_w - x_offset)
    h = min(fg_h, bg_h, fg_h + y_offset, bg_h - y_offset)

    if w < 1 or h < 1: return

    # clip foreground and background images to the overlapping regions
    bg_x = max(0, x_offset)
    bg_y = max(0, y_offset)
    fg_x = max(0, x_offset * -1)
    fg_y = max(0, y_offset * -1)
    foreground = foreground[fg_y:fg_y + h, fg_x:fg_x + w]
    background_subsection = background[bg_y:bg_y + h, bg_x:bg_x + w]

    # separate alpha and color channels from the foreground image
    foreground_colors = foreground[:, :, :3]
    alpha_channel = foreground[:, :, 3] / 255  # 0-255 => 0.0-1.0

    # construct an alpha_mask that matches the image shape
    alpha_mask = np.dstack((alpha_channel, alpha_channel, alpha_channel))

    # combine the background with the overlay image weighted by alpha
    composite = background_subsection * (1 - alpha_mask) + foreground_colors * alpha_mask

    # overwrite the section of the background image that has been updated
    background[bg_y:bg_y + h, bg_x:bg_x + w] = composite

def readFrames():
    global tokens
    global danger
    global drawing
    
    # Defaults to 500
    history = 100
    # Defaults to 16
    varThreshold = 16
    detectShadows = True
    backSub = cv2.createBackgroundSubtractorMOG2(history, varThreshold, detectShadows)
    kernel=np.ones((5,5),np.uint8)

    while(1):
        ret, frame = cap.read()
        frame = cv2.flip(frame, 1)
        img_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        fgMask = backSub.apply(img_gray)
        # Try to remove single pixels using erosion
        fgMask=cv2.erode(fgMask,kernel,iterations=1)
        img_gray = cv2.bitwise_and(img_gray, img_gray, mask=fgMask)

        # Detect edges
        edges = cv2.Canny(img_gray, 100, 200)

        # Detect the contours on the binary image using cv2.CHAIN_APPROX_NONE
        contours, hierarchy = cv2.findContours(image=edges, mode=cv2.RETR_TREE, method=cv2.CHAIN_APPROX_NONE)
                                            
        # Draw contours on the original image
        image_copy = frame.copy()
        #cv2.drawContours(image=image_copy, contours=contours, contourIdx=-1, color=(0, 255, 0), thickness=2, lineType=cv2.LINE_AA)
                    
        # see the results
        #cv2.imshow('None approximation', image_copy)

        contours_poly = [None]*len(contours)
        boundRect = [None]*len(contours)
        centers = [None]*len(contours)
        radius = [None]*len(contours)
        for i, c in enumerate(contours):
            contours_poly[i] = cv2.approxPolyDP(c, 3, True)
            boundRect[i] = cv2.boundingRect(contours_poly[i])
            centers[i], radius[i] = cv2.minEnclosingCircle(contours_poly[i])
        
        lock.acquire()
        #drawing = np.zeros((edges.shape[0], edges.shape[1], 3), dtype=np.uint8)
        drawing = image_copy

        for i in range(len(contours)):
            #cv2.rectangle(drawing, (int(boundRect[i][0]), int(boundRect[i][1])), \
            #(int(boundRect[i][0]+boundRect[i][2]), int(boundRect[i][1]+boundRect[i][3])), color, 2)

            # Check for collisions
            for token in tokens:
                player_rect = Rect(token.x, token.y, imgW, imgH)
                player_rect2 = Rect(int(boundRect[i][0]), int(boundRect[i][1]), int(boundRect[i][2]), int(boundRect[i][3]))
                collide = pygame.Rect.colliderect(player_rect, player_rect2)
                if collide:
                    playsound('pling.mp3', block = False)
                    game.points = game.points + 1
                    token.x = random.randint(0, screenWidth-imgW)
                    token.y = 0
                    token.speed = random.randint(1,maxSpeed)
        
            if danger != None:
                player_rect = Rect(danger.x, danger.y, imgW, imgH)
                player_rect2 = Rect(int(boundRect[i][0]), int(boundRect[i][1]), int(boundRect[i][2]), int(boundRect[i][3]))
                collide = pygame.Rect.colliderect(player_rect, player_rect2)
                if collide:
                    playsound('explosion.wav', block = False)
                    game.points = game.points - 10
                    if game.points < 0:
                        game.points = 0
                    danger.x = random.randint(0, screenWidth-imgW)
                    danger.y = 0


        lock.release()

def drawCircles():
    global tokens
    global danger
    global drawing

    while(1):

        time.sleep(5/1000)

        if drawing is None:
            continue

        h, w, _ = drawing.shape

        for token in tokens:
            token.y = token.y + token.speed
            if (token.y + imgH > h):
                if (game.points > 0):
                    game.points = game.points - 1
                token.y = 0

        if danger != None:
            danger.y = danger.y + danger.speed
            if (danger.y + imgH > h):
                danger = None
                #game.points = 0


def updateTime():
    game.time = game.time + 1
    threading.Timer(1, updateTime).start()

def updateDanger():
    global danger
    global screenWidth
    if screenWidth != 0:
        danger = Danger(random.randint(0, screenWidth-imgW))
    threading.Timer(10, updateDanger).start()

if __name__ == "__main__":
    readFramesThread = Thread(target = readFrames, args = ())
    readFramesThread.start()

    drawCirclesThread = Thread(target = drawCircles, args = ())
    drawCirclesThread.start()

    updateTime()
    updateDanger()

    # IMREAD_UNCHANGED => open image with the alpha channel
    coin = cv2.imread("coin.png", cv2.IMREAD_UNCHANGED)
    coin = cv2.resize(coin, (imgH, imgW))
    rocket = cv2.imread("rocket.png", cv2.IMREAD_UNCHANGED)
    rocket = cv2.resize(rocket, (imgH, imgW))

    while(True):
        # Capture the video frame by frame
        #ret, frame = cap.read()

        if drawing is None:
            continue

        # Initialize
        if len(tokens) == 0:
            h, w, _ = drawing.shape
            screenWidth = w
            positionText = (w - 100, 50)
            positionGameTimeText = (10, 50)
            for x in range(tokenCount):
                tokens.append(Token(random.randint(0, w-imgW), 0, random.randint(1, maxSpeed)))

        lock.acquire()
        #cv2.circle(drawing, tuple([100, imgY]), 20, color, -1)
        
        #added_image = cv2.addWeighted(drawing[imgY:imgY+imgH,imgX:imgX+imgW,:],0,rocket[0:imgH,0:imgW,:],1,0)
        #drawing[imgY:imgY+imgH,imgX:imgX+imgW] = added_image
        for token in tokens:
            add_transparent_image(drawing, coin, token.x, token.y)

        if danger != None:
            add_transparent_image(drawing, rocket, danger.x, danger.y)

        cv2.putText(drawing, str(game.points), positionText, cv2.FONT_HERSHEY_SIMPLEX, #font family
            1, #font size
            colorText, 2) #font stroke

        gameTime = str(math.floor(game.time/60)) + ":" + str(game.time % 60)
        cv2.putText(drawing, gameTime, positionGameTimeText, cv2.FONT_HERSHEY_SIMPLEX, #font family
            1, #font size
            colorText, 2) #font stroke

        #rocket = cv2.imread("image.jpg")
        #cv2.imshow('frame', rocket)
        cv2.imshow('frame', drawing)
        lock.release()
        
        #fps = cap.get(cv2.CAP_PROP_FPS)
        if cv2.waitKey(1) & 0xFF == ord('n'):
            game.points = 0
            game.time = 0

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    # After the loop release the cap object
    cap.release()
    # Destroy all the windows
    cv2.destroyAllWindows()