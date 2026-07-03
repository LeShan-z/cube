import time
import sys
import cv2
import serial
#import socket
import numpy as np

from color_recognition_algorithm import color_classification
from twophase import solve
#import matplotlib.pyplot as plt
#import asyncio

def outline_frame(cam, height, width):
    _, frame = cam.read()
    print(frame.shape)
    overlay = np.zeros((height, width, 3), np.uint8)
    frame_with_rec = cv2.addWeighted(frame, 0.15, overlay, 0.1, 0)
    x, y = int(width / 2), int(height / 2)
    cv2.rectangle(frame_with_rec, (x - 240, y - 240), (x + 240, y + 240), (0, 0, 255), 5)
    frame_with_rec = cv2.flip(frame_with_rec, 1)

    cube_frame = frame[y - 240:y + 240, x - 240:x + 240]#jhy:change offset 
    frame_with_rec[y - 240:y + 240, x - 240:x + 240] = cv2.flip(cube_frame, 1)
    # cv2.GaussianBlur(cube_frame, (5, 5), 0)

    return frame, frame_with_rec, cube_frame


myserial = serial.Serial("/dev/ttyUSB0", 115200, timeout=0.5)


def Light_Open():
    print("open light")
    myserial.write(b'#X!') 
    return 1
def Light_Close():
    print("close light")
    myserial.write(b'#Y!')
    return 0



def camp_set_time(cam,height, width,times):
    
    for i in range(times):
        frame,  frame_with_rec, cube_frame = outline_frame(cam, height, width)
        cv2.imshow("frame", frame_with_rec)
        cv2.imshow("cube_frame", cube_frame)
    


def recognize_cube(cam, myserial):
    _, frame = cam.read()
    
    height, width, _ = frame.shape
    
    light=0
    side_flag = 1
    auto_start_once = True
    while True:

        # 确认魔方在画面中位置，需调整
        frame,  frame_with_rec, cube_frame = outline_frame(cam, height, width)
     
        cv2.imshow("frame", frame_with_rec)
        cv2.imshow("cube_frame", cube_frame)
        k = cv2.waitKey(1) & 0xFF
        if auto_start_once:
            auto_start_once = False
            k = ord("s")
        if k == ord('l'):
            light = Light_Open()
        if k == ord('k'):
            light = Light_Close()
           #if light == 0:
           #    light = Light_Open()
           #elif light == 1:
           #    light = Light_Close()

        if k == ord('s'):
            print("1")
            cv2.imwrite("cube_test" + str(side_flag) + ".jpg", cube_frame)
            time.sleep(0.5)
            myserial.write(b'#BU!')
            camp_set_time(cam, height, width,80)
            print("2")
            

      
            
            
            side_flag += 1
            frame,  frame_with_rec, cube_frame = outline_frame(cam, height, width)
            cv2.imshow("cube_frame", cube_frame)
            cv2.imshow("frame", frame_with_rec)
           
            cv2.imwrite("cube_test" + str(side_flag) + ".jpg", cube_frame)
             
            myserial.write(b'#BUM!')
            camp_set_time(cam, height, width,80)
          
  
            
            
            side_flag += 1
            frame,  frame_with_rec, cube_frame = outline_frame(cam, height, width)
            cv2.imshow("cube_frame", cube_frame)
            cv2.imshow("frame", frame_with_rec)
            
            cv2.imwrite("cube_test" + str(side_flag) + ".jpg", cube_frame)
           
            myserial.write(b'#MBU!')
            camp_set_time(cam, height, width,80)
              
  
            
            
            side_flag += 1
            frame,  frame_with_rec, cube_frame = outline_frame(cam, height, width)
            cv2.imshow("cube_frame", cube_frame)
            cv2.imshow("frame", frame_with_rec)
           
            cv2.imwrite("cube_test" + str(side_flag) + ".jpg", cube_frame)
           
            myserial.write(b'#IBU!')
            camp_set_time(cam, height, width,80)
                
            
            side_flag += 1
            frame,  frame_with_rec, cube_frame = outline_frame(cam, height, width)
            cv2.imshow("cube_frame", cube_frame)
            cv2.imshow("frame", frame_with_rec)
            
            cv2.imwrite("cube_test" + str(side_flag) + ".jpg", cube_frame)
            
            myserial.write(b'#BBUM!')
            camp_set_time(cam, height, width,80)
                         
    
       
            side_flag += 1
            frame,  frame_with_rec, cube_frame = outline_frame(cam, height, width)
            cv2.imshow("cube_frame", cube_frame)
            cv2.imshow("frame", frame_with_rec)
            cv2.imwrite("cube_test" + str(side_flag) + ".jpg", cube_frame)
          

            print("save" + str(side_flag)+ ".jpg successfuly!")
            side_flag += 1
            print("-------------------------")
        elif k == 27:
            break
        elif side_flag == 7:
            break
        elif k == ord('='): 
            print("light+")
            myserial.write(b'#=!')
        elif k == ord('-'):
            print("light-")
            myserial.write(b'#-!')

    time.sleep(1)  # 休眠5秒

    # 颜色分类，返回每一面从左上到右下的颜色，面的顺序为F U B D R L
    color_pred = color_classification()

    F = color_pred[0:9]
    U = color_pred[9:18]
    B = color_pred[18:27]
    D = color_pred[27:36]
    R = color_pred[36:45]
    L = color_pred[45:54]

    # 求解函数输入格式应按U, R, F, D, L, B
    temp = np.append(U, R)
    temp = np.append(temp, F)
    temp = np.append(temp, D)
    temp = np.append(temp, L)
    cube_str_temp = np.append(temp, B)

    cube_str = ''
    for i in range(0, 54):
        cube_str += cube_str_temp[i]
    print(cube_str, len(cube_str))
    try:
        solution = solve(cube_str)
    except:
        solution = 'not found'

    return solution


def destroy_windows():
    windows = ('frame', 'cube_frame')
    for window in windows:
        cv2.destroyWindow(window)


def exec_0to5():
    for i in range(6):
        if cur[i] == 0:
            cur[i] = 5
        elif cur[i] == 1:
            cur[i] = 4
        elif cur[i] == 2:
            cur[i] = 2
        elif cur[i] == 3:
            cur[i] = 3
        elif cur[i] == 4:
            cur[i] = 0
        elif cur[i] == 5:
            cur[i] = 1


def exec_1to5():
    for i in range(6):
        if cur[i] == 0:
            cur[i] = 4
        elif cur[i] == 1:
            cur[i] = 5
        elif cur[i] == 2:
            cur[i] = 2
        elif cur[i] == 3:
            cur[i] = 3
        elif cur[i] == 4:
            cur[i] = 1
        elif cur[i] == 5:
            cur[i] = 0


def exec_2to5():
    for i in range(6):
        if cur[i] == 0:
            cur[i] = 3
        elif cur[i] == 1:
            cur[i] = 2
        elif cur[i] == 2:
            cur[i] = 5
        elif cur[i] == 3:
            cur[i] = 4
        elif cur[i] == 4:
            cur[i] = 0
        elif cur[i] == 5:
            cur[i] = 1


def exec_3to5():
    for i in range(6):
        if cur[i] == 0:
            cur[i] = 2
        elif cur[i] == 1:
            cur[i] = 3
        elif cur[i] == 2:
            cur[i] = 4
        elif cur[i] == 3:
            cur[i] = 5
        elif cur[i] == 4:
            cur[i] = 0
        elif cur[i] == 5:
            cur[i] = 1


def exec_4to5():
    for i in range(6):
        if cur[i] == 0:
            cur[i] = 1
        elif cur[i] == 1:
            cur[i] = 0
        elif cur[i] == 2:
            cur[i] = 2
        elif cur[i] == 3:
            cur[i] = 3
        elif cur[i] == 4:
            cur[i] = 5
        elif cur[i] == 5:
            cur[i] = 4


def step_transformation():
    global outcome
    str_outcome = outcome
    step_count = 0
    step = []
    i = 0
    while i < len(str_outcome):
        if i==len(str_outcome)-1:
            ch = [str_outcome[i],' ']
        else:
            ch = [str_outcome[i], str_outcome[i + 1]]
        if ch[1] == ' ':
            i += 2
            t_ch = 3
        elif ch[1] == '2':
            i += 3
            t_ch = 2
        else:
            i += 3
            t_ch = 1

        idx = s[ch[0]]
        if cur[idx] != 5:
            if cur[idx] == 0:
                exec_0to5()
                step.append(5)
                step_count += 1
                # print("上翻一次\n")
            elif cur[idx] == 1:
                exec_1to5()
                step.append(4)
                step_count += 1
                # print("上翻三次\n")
            elif cur[idx] == 2:
                exec_2to5()
                step.append(7)
                step_count += 1
                # print("逆转90然后上翻一次\n")
            elif cur[idx] == 3:
                exec_3to5()
                step.append(6)
                step_count += 1
                # print("顺转90然后上翻一次\n")
            elif cur[idx] == 4:
                exec_4to5()
                step.append(8)
                step_count += 1
                # print("上翻两次\n")
        step.append(t_ch)
        step_count += 1
        # print(cur)
    outcome = ""
    return step


def step_run(step):
    last_step = 0
    next_step = 0
    res = "#BUJ"
    temp_state = "ok"
    for j in range(len(step)):
        if j > 0:
            last_step = step[j - 1]
        else:
            last_step = 0
        if j + 1 < len(step):
            next_step = step[j + 1]
        else:
            next_step = 0
        if step[j] == 1:
            if temp_state == "locked_rotor":
                return "locked_rotor"
            res += "R"
        elif step[j] == 2:
            if temp_state == "locked_rotor":
                return "locked_rotor"
            res += "A"
        elif step[j] == 3:
            if temp_state == "locked_rotor":
                return "locked_rotor"
            res += "L"
        elif step[j] == 4:
            if temp_state == "locked_rotor":
                return "locked_rotor"
            res += "BBB"
        elif step[j] == 5:
            if temp_state == "locked_rotor":
                return "locked_rotor"
            res += "B"
        elif step[j] == 6:
            if temp_state == "locked_rotor":
                return "locked_rotor"
            res += "JB"
        elif step[j] == 7:
            if temp_state == "locked_rotor":
                return "locked_rotor"
            res += "IB"
        elif step[j] == 8:
            if temp_state == "locked_rotor":
                return "locked_rotor"
            res += "BB"
    res += "U!"
    return res


# def main():
#     step = step_transformation()
#     res = step_run(step)
#     print(res)


cur = [0] * 10
outcome = ""  # 结算得到的结果

s = {'F': 0, 'B': 1, 'L': 2, 'R': 3, 'U': 4, 'D': 5}


def main():
    cam = cv2.VideoCapture(0)
    if not cam.isOpened():
        raise Exception("Failed to open the camera.")
    
    solution = recognize_cube(cam, myserial)
    # solution = recognize_cube()
    print(solution)
    for i in range(6):
        cur[i] = i
    global outcome
    outcome = solution
    step = step_transformation()
    res = step_run(step)
    print("机器指令：" + res)
    myserial.write(res.encode())
    cam.release()
    destroy_windows()
    #input("Execute success! Any key to quit!")

if __name__ == "__main__":
    main()
                           
