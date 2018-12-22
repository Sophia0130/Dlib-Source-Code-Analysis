## Dlib源码解读

### 1.FHOG特征提取和SVM训练

http://dlib.net/fhog_object_detector_ex.cpp.html?tdsourcetag=s_pctim_aiomsg

参考官网给的demo

### 2.目标检测

主要是对 evaluate_detectors() 函数的解析，和FHOG特征提取的理解

函数的调用的层次结构   

![](https://github.com/Sophia0130/Dlib-Source-Code-Analysis/blob/master/%E5%87%BD%E6%95%B0%E8%A7%A3%E6%9E%90.png?raw=true)

用到的函数和类所在的头文件     

![](https://github.com/Sophia0130/Dlib-Source-Code-Analysis/blob/master/%E5%A4%B4%E6%96%87%E4%BB%B6%E8%A7%A3%E6%9E%90.png?raw=true)