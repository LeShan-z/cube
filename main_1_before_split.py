import time
import sys
import cv2
import math
import serial
#import socket
import numpy as np

from twophase import solve
#import matplotlib.pyplot as plt
#import asyncio

# face_coordinates = np.array([[(12, 15), (75, 70)], [(90, 15), (152, 70)], [(170, 15), (230, 70)],
#                              [(12, 90), (75, 145)], [(90, 90), (152, 145)], [(170, 90), (230, 145)],
#                              [(12, 165), (75220)], [(90, 165), (152, 220)], [(170, 165), (230, 220)]])
#face_coordinates = np.array([[(25, 25), (55, 55)], [(105, 25), (135, 55)], [(185, 25), (215, 55)],
#                             [(25, 105), (55, 135)], [(105, 105), (135, 135)], [(185, 105), (215, 135)],
#                             [(25, 185), (55, 215)], [(105, 185), (135, 215)], [(185, 185), (215, 215)]])



face_coordinates = np.array([[(50, 50), (110, 110)], [(210, 50), (270, 110)], [(370, 50), (430, 110)],
                             [(50, 210), (110, 270)], [(210, 210), (270, 270)], [(370, 210), (430, 270)],
                             [(50, 370), (110, 430)], [(210, 370), (270, 430)], [(370, 370), (430, 430)]])

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


def maxmin3(a, b, c):
    if a < b:
        max = b
        min = a
    else:
        max = a
        min = b
    if max < c:
        max = c
    if min > c:
        min = c
    return max, min

def RGB2HSV(RGB_feature):
    x, y = RGB_feature.shape
    HSV_feature = np.zeros((x,y))

    for i in range(0, x):
        r = RGB_feature[i, 0] / 255
        g = RGB_feature[i, 1] / 255
        b = RGB_feature[i, 2] / 255
        max, min = maxmin3(r, g, b)
        delta = (max - min)

        if delta == 0:
            HSV_feature[i, 0] = 0
        else:
            if r == max:
                HSV_feature[i, 0] = ((g - b) / delta) * 60
            elif g == max:
                HSV_feature[i, 0] = 120 + (((b - r) / delta) * 60)
            elif b == max:
                HSV_feature[i, 0] = 240 + (((r-g)/delta)*60)
            if HSV_feature[i, 0] < 0:
                HSV_feature[i, 0] += 360
        if max == 0:
            HSV_feature[i, 1] = 0
        else:
            HSV_feature[i, 1] = delta / max

        HSV_feature[i, 2] = max

    return HSV_feature

def cluster(HSV_Feature):
    index = np.zeros(54, dtype=int)
    for i in range(0, 54):
        index[i] = i

    #寻找白色，挑选饱和度最低的9个色块，将其放在index[45] - index[53]
    #使用冒泡排序将最低的9个放在index最后
    for i in range(0, 9):
        for j in range(0, 54-i-1):
            if HSV_Feature[index[j]][1] < HSV_Feature[index[j+1]][1]:
                tmp = index[j]
                index[j] = index[j+1]
                index[j+1] = tmp

    white_center_index = -1
    #检测是否有白色中心块，若无则说明颜色错分
    for i in range(45, 54):
        if index[i] % 9 == 4:
            white_center_index = index[i]
            print("白色中心块序号为： " + str(white_center_index))

    if white_center_index == -1:
        print("未找到白色中心块，请重新拍照")

    #1.选择魔方的中心块颜色作为初始的聚类中心
    h_center = np.zeros(5)
    h_center_old = np.array([-1, 0, 0, 0, 0],dtype=float)
    group = np.zeros((5, 45), dtype=int)
    group_count = np.zeros(5, dtype=int)
    h_center_index = 0
    for i in range(0, 6):
        color_index = 9 * i + 4
        if color_index != white_center_index:
            h_center[h_center_index] = HSV_Feature[color_index][0]
            h_center_index += 1

    #2.计算其他颜色到聚类中心的距离,划分到最接近的
    MAX_LOOP = 50
    for loop in range(0, MAX_LOOP):
        for j in range(0, 5):
            group_count[j] = 0
        for i in range(0, 45):
            # index[i]0 - 44没有白色色块, 45 - 53全是白色色块
            h = HSV_Feature[index[i]][0]
            min_dist = float("inf")
            min_dist_group = 0
            for j in range(0, 5):
                h0 = h_center[j]
                #要计算这两个角度之间的绝对值差，首先需要处理两个角度间的差值使其落在0到360度之间，
                #因为简单地做差可能得出的结果会超过这个范围，例如当h0 = 350而h = 10时，它们的差应该是20度而不是340度。
                diff = abs(h - h0) % 360 #先求差值并取绝对值，然后取模360度
                if diff > 180: #如果差值大于180度，则补足360度
                    dist = 360-diff
                else:
                    dist = diff
                if dist < min_dist:
                    min_dist = dist
                    min_dist_group  = j
            group[min_dist_group][group_count[min_dist_group]] = index[i]
            group_count[min_dist_group] += 1

        #3.重新计算每个聚类中所有点的平均值,作为新的中心
        for j in range(0, 5):
            sum_cos = 0
            sum_sin = 0
            pi = math.pi
            for i in range(0, group_count[j]):
                hsv_color_index = group[j][i]
                a = HSV_Feature[hsv_color_index][0] * (pi / 180) #计算弧度
                sum_cos += math.cos(a)
                sum_sin += math.sin(a)

            if group_count[j] > 0:
                a = math.atan2(sum_sin, sum_cos) * (180 / pi)
                h_center[j] = (a + 360) % 360

        #4.检查聚类中心是否变化
        change = 0
        for j in range(0, 5):
            if h_center_old[j] != h_center[j]:
                change += 1

        if change == 0:
            print("聚类中心已经没有改变, 循环： " + str(loop))
            break
        else:
            for j in range(0, 5):
                h_center_old[j] = h_center[j]

    #将聚类变为字符串
    #面的顺序为F U B D R L
    color_str = np.array(['F', 'U', 'B', 'D', 'R', 'L'])
    color_pred = np.empty(54, dtype=str)
    group_count_index = 0
    for i in range(0, 6):
        center_index = 9 * i + 4
        if center_index != white_center_index:
            if group_count[group_count_index] != 9:
                print("分类错误(某类色块个数有误), 调整灯光, 重新拍照")

            center_in_group = 0
            for j in range(0, group_count[group_count_index]):
                index = group[group_count_index][j]
                color_pred[index] = color_str[i]
                if index == center_index:
                    center_in_group += 1
            if center_in_group != 1:
                print("分类错误(某面中心块错分类), 调整灯光, 重新拍照")

            group_count_index += 1

    #填入白色
    for i in range(0, 54):
        if color_pred[i] == '':
            color_pred[i] = color_str[int((white_center_index-4) / 9)]

    return color_pred


def color_classification():
    RGB_feature_temp = []
    HS_feature_temp = []
    for i in range(1, 7):
        image = cv2.imread('cube_test%d.jpg' % i)
        im_RGB = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        # 将每个小块分割
        for cordinates in face_coordinates:
            face_RGB = im_RGB[cordinates[0, 1]:cordinates[1, 1], cordinates[0, 0]:cordinates[1, 0]]
            # cv2.imshow("face", cv2.cvtColor(face_RGB, cv2.COLOR_RGB2BGR))
            # cv2.waitKey()

            # 计算每个小块三通道的BGR均值，注意这里均值出来是四维
            temp_rgb = cv2.mean(face_RGB)
            temp_rgb= temp_rgb[0:3]
            RGB_feature_temp.append(temp_rgb)


    RGB_feature = np.array(RGB_feature_temp)

    #将rgb转为hsv，H：0-360， S、V：0-1
    HSV_feature = RGB2HSV(RGB_feature)

    #聚类
    color_pred = cluster(HSV_feature)

    return color_pred

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
                           