

# -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- --

#region : File Description and Imports
"""
image_interpretation.py
Skills activity code for image interpretation lab guide.
Students will perform camera calibration along with line detection.
Please review Lab Guide - Image Interpretation PDF
"""
from pal.products.qcar import QCarCameras,QCarRealSense, IS_PHYSICAL_QCAR
from hal.utilities.image_processing import ImageProcessing
import time
import numpy as np
import cv2
#endregion


#----INICIO------------------------------------------------------------------------


#import numpy as np
from scipy.special import logit, expit
from scipy import ndimage
from threading import Thread, Lock
import time
import pyqtgraph as pg
import signal

from pal.products.qcar import QCar, QCarGPS, IS_PHYSICAL_QCAR
from pal.utilities.scope import MultiScope 
from pal.utilities.math import find_overlap, wrap_to_2pi, wrap_to_pi
from hal.products.qcar import QCarEKF, QCarDriveController
from hal.products.mats import SDCSRoadMap

#----------------------LIBRERIAS VEHICULE CONTROL------------------------------------

import os
import signal
import numpy as np
from threading import Thread
import time
import cv2
import pyqtgraph as pg

from pal.products.qcar import QCar, QCarGPS, IS_PHYSICAL_QCAR
from pal.utilities.math import wrap_to_pi
from hal.products.qcar import QCarEKF
from hal.products.mats import SDCSRoadMap
import pal.resources.images as images

from hal.utilities.image_processing import ImageProcessing
import numpy as np
import cv2
from pal.products.qcar import QCarRealSense
from QCarRealSense_EX import QCarRealSense_EX
#------------------------------------

tf = 60
startDelay = 1
controllerUpdateRate = 380
# ===== Vehicle Controller Parameters
# - enableVehicleControl: If true, the QCar will drive through the specified
#   node sequence. If false, the QCar will remain stationary.
# - v_ref: desired velocity in m/s
# - nodeSequence: list of nodes from roadmap. Used for trajectory generation.
#enableVehicleControl = True
#_ref = 0.32

lowerBounds_amarillo = (20, 100, 100)
upperBounds_amarillo = (40, 255, 255)
lowerBounds_lightRed = (0,100,20)
upperBounds_lightRed = (5, 255, 255)
#---------------------CODIGO DE VELOCIDAD--------------------
v_ref = 0.45
K_p = 0.1
K_i = 1   # ORIGINALMENTE 1


# ===== Steering Controller Parameters
# - enableSteeringControl: whether or not to enable steering control
# - K_stanley: K gain for stanley controller
# - nodeSequence: list of nodes from roadmap. Used for trajectory generation.
enableSteeringControl = True
K_stanley = 1 #Mejor performance= 1
Kd_stanley = 11 #Mejor performance =11

#nodeSequence = [0,20,0]

#nodeSequence = [9,14,18,9]

#nodeSequence = [10,2,4,13,14,6,22,10]

nodeSequence = [10,9,21,2,4,13,14,6,20,8,22,10]

##nodeSequence = [-13.093,-7.572,0.005,0,0]


# ===== Occupancy Grid Parameters
# - cellWidth: edge length for occupancy grid cells (in meters)
# - r_res: range resolution for polar grid cells (in meters)
# - r_max: maximum range of the lidar (in meters)
# - p_low: likelihood value for vacant cells (according to lidar scan)
# - p_high: likelihood value for occupied cells (according to lidar scan)
#
cellWidth = 1.25
r_res = 0.05
r_max = 4
p_low = 0.4
p_high = 0.6


#region Initial Setup
lock = Lock()

#region : Initial setup

#region : Initial setup



if enableSteeringControl:
    roadmap = SDCSRoadMap(leftHandTraffic=False)
    waypointSequence = roadmap.generate_path(nodeSequence)
    initialPose = roadmap.get_node_pose(nodeSequence[0]).squeeze()   
    #x_hat = initialPose
    #t_hat = 0
else:
    initialPose = [0, 0, 0]



if not IS_PHYSICAL_QCAR:
    import Setup_Competition
    Setup_Competition.setup( #bsa-> Yo agregué el qqcar
        initialPosition=[initialPose[0], initialPose[1], 0],
        initialOrientation=[0, 0, initialPose[2]]
    )

# Used to enable safe keyboard triggered shutdown
global KILL_THREAD
KILL_THREAD = False
def sig_handler(*args):
    global KILL_THREAD
    KILL_THREAD = True
signal.signal(signal.SIGINT, sig_handler)
#endregion


#gps = QCarGPS(initialPose=initialPose)
#while (not KILL_THREAD) and (gps.readGPS() or  gps.readLidar()):
 #   pass


#---------------------------------CLASES Y OBJETOS DEL CONTROLADOR DE VELOCIDAD--------------------
#endregion



#----------------------------LISTO---------INICIO---------------------------------------------------------------------------------
class SpeedController:

    def __init__(self, kp=0, ki=0):
        self.maxThrottle = 0.9

        self.kp = kp
        self.ki = ki
        
        self.ei = 0



    # ==============  SECTION A -  Speed Control  ====================
    def update(self, v, v_ref, dt):
        #
        e = v_ref - v
        self.ei += dt*e

        return np.clip(
            self.kp*e + self.ki*self.ei,
            -self.maxThrottle,
            self.maxThrottle
        )
        
        #return 0


class SteeringController:

    def __init__(self, waypoints, k=1, k_d=0.1, cyclic=False):
        self.maxSteeringAngle = np.pi/6

        self.wp = waypoints
        self.N = len(waypoints[0, :])
        self.wpi = 0

        self.k = k
        self.cyclic = cyclic

        self.p_ref = (0, 0)
        self.th_ref = 0

        #Esto ya lo agregué yo
        self.k_d = k_d  # Ganancia derivativa
        self.pe = 0

        #region BSA-> Intento
        self.nwp_IS_A_NODE = False
       
        #endregion
        
        


    # ==============  SECTION B -  Steering Control  ====================
    def update(self, p, th, speed):
        wp_1 = self.wp[:, np.mod(self.wpi, self.N-1)]
        wp_2 = self.wp[:, np.mod(self.wpi+1, self.N-1)]
        wp_20 = self.wp[:, np.mod(self.wpi+20, self.N-1)]
     
        v = wp_2 - wp_1

        #region BSA-> Intento de desacelerar


        #--------------------PRIMER SEMAFORO------------------
        self.nwp_IS_A_NODE = False
        if(wp_20 == self.wp[:, 440]).all(): 
            self.nwp_IS_A_NODE = True
        if(wp_20 == self.wp[:, 720]).all(): 
            self.nwp_IS_A_NODE = True
        if(wp_20 == self.wp[:, 1150]).all(): 
            self.nwp_IS_A_NODE = True #500 deberia ser el waypoint del nodo
        if(wp_20 == self.wp[:, 1490]).all(): 
            self.nwp_IS_A_NODE = True
        

        
            #1110 SEGUNDO ALTO
            #685 PRIMER ALTO
            #440 primer semaforo
        

        #-------------FIN PRIMER SEMAFORO----------------------

        #-------------------FIN SEGUNDO ALTO------------------------------------
        #
        #endregion
        #AQUI


        #endregion
                
        v_mag = np.linalg.norm(v)
        try:
            v_uv = v / v_mag
        except ZeroDivisionError:
            return 0

        tangent = np.arctan2(v_uv[1], v_uv[0]) #Este tangent da un ángulo en radianes

        s = np.dot(p-wp_1, v_uv)

        if s >= v_mag:
            if  self.cyclic or self.wpi < self.N-2:
                self.wpi += 1

        ep = wp_1 + v_uv*s
        ct = ep - p
        dir = wrap_to_pi(np.arctan2(ct[1], ct[0]) - tangent)

        ect = np.linalg.norm(ct) * np.sign(dir)
        psi = wrap_to_pi(tangent-th)

        self.p_ref = ep
        
        self.th_ref = tangent

        #Esto lo agregué para el control derivativo
        delta_error = psi - self.pe
        self.pe = psi

        #Previous vector
        self.ve = v

        return np.clip(
            wrap_to_pi(psi + np.arctan2(self.k*ect+ self.k_d * delta_error , speed)), #+ self.k_d * delta_error esto lo agregué yo
            -self.maxSteeringAngle,
            self.maxSteeringAngle)
        #-->
        return 0





#----------------------------LISTO---------FIN---------------------------------------------------------------------------------


#-----------------NO SE USA Y NO SE PIENSA USAR -----------------------------------------------------------------------------
class OccupancyGrid:

    def __init__(self,
            x_min=-4,
            x_max=3,
            y_min=-3,
            y_max=6,
            cellWidth=0.02,
            r_max=5,
            r_res=0.02,
            p_low=0.4,
            p_high=0.6
        ):

        #region define probabilities and their log-odds forms
        self.p_low = p_low
        self.p_prior = 0.5
        self.p_high = p_high
        self.p_sat = 0.001

        self.l_low = logit(self.p_low)
        self.l_prior = logit(self.p_prior)
        self.l_high = logit(self.p_high)
        self.l_min = logit(self.p_sat)
        self.l_max = logit(1-self.p_sat)
        #endregion

        self.init_polar_grid(r_max, r_res)

        self.init_world_map(
            x_min=x_min,
            x_max=x_max,
            y_min=y_min,
            y_max=y_max,
            cellWidth=cellWidth
        )
        self.init_patch()

    # ==============  SECTION A - Polar Grid ====================
    def init_polar_grid(self, r_max, r_res):
        # Configuration Parameters for polar grid
        fov = 2*np.pi
        self.phiRes = 1 * np.pi/180
        self.r_max = r_max
        self.r_res = r_res

        # Size of polar patch
        self.mPolarPatch = np.int_(np.ceil(fov / self.phiRes))
        self.nPolarPatch = np.int_(np.floor(self.r_max/self.r_res))


        self.polarPatch = np.zeros(
            shape = (self.mPolarPatch, self.nPolarPatch),
            dtype = np.float32
        )

    def update_polar_grid(self, r):
        # Implement code here to populate the values of self.polarPatch
        # given LiDAR range data 'r'.
        # - r is a 1D list of length self.mPolarPatch
        # - All range measurements are equally self.phiRes radians apart,
        #   starting with 0

        # Implement Your Solution Here

        pass

        # Quanser Solution Implemented Below
        r = np.int_(np.round(r / self.r_res))
        for i in range(self.mPolarPatch):
            if r[i] > 0:
                self.polarPatch[i,:r[i]] = self.l_low
                self.polarPatch[i,r[i]:r[i]+1] = self.l_high
                self.polarPatch[i,r[i]+1:] = self.l_prior
            else:
                self.polarPatch[i,:] = self.l_prior
        #'''


    # ==============  SECTION B - Interpolation ====================
    def init_patch(self):
        self.nPatch = np.int_(2*np.ceil(self.r_max/self.cellWidth) + 1)
        self.patch = np.zeros(
            shape = (self.nPatch, self.nPatch),
            dtype = np.float32
        )

    def generate_patch(self, th):
        # Implement Your Solution Here
        pass

        # Quanser Solution Implemented Below

        cx = (self.nPatch * self.cellWidth) / 2
        cy = cx

        x = np.linspace(-cx, cx, self.nPatch)
        y = np.linspace(-cy, cy, self.nPatch)
        xv, yv = np.meshgrid(x, y)

        rPatch = (
            np.sqrt(np.square(xv) + np.square(yv)) / self.r_res
        )
        phiPatch = (
            wrap_to_2pi(np.arctan2(yv, xv) + th) / self.phiRes
        )
        ndimage.map_coordinates(
            input=self.polarPatch,
            coordinates=[phiPatch, rPatch],
            output=self.patch
        )
        ##'''

    # ==============  SECTION C - Occupancy Grid Update  ====================
    def init_world_map(self,
            x_min = -4,
            x_max = 3,
            y_min = -3,
            y_max = 6,
            cellWidth=0.02
        ):

        self.x_min = x_min
        self.x_max = x_max
        self.y_min = y_min
        self.y_max = y_max
        self.cellWidth = cellWidth
        self.xLength = x_max - x_min
        self.yLength = y_max - y_min
        self.m = np.int_(np.ceil(self.yLength/self.cellWidth))
        self.n = np.int_(np.ceil(self.xLength/self.cellWidth))

        self.map = np.full(
            shape = (self.m, self.n),
            fill_value = self.l_prior,
            dtype = np.float32
        )

    def xy_to_ij(self, x, y):
        i = np.int_(np.round( (self.y_max - y) / self.cellWidth ))
        j = np.int_(np.round( (x - self.x_min) / self.cellWidth ))
        return i, j

    def updateMap(self, x, y, th, angles, distances):

        # Function created in SECTION A
        self.update_polar_grid(distances)

        # Function created in SECTION B
        self.generate_patch(th)
        # Implement code here to update self.map using self.patch

        pass

        # Quanser Solution Implemented Below
        # calculate Location of cells to update position offsets
        iy, jx = self.xy_to_ij(x, y)

        iTop = np.int_(iy - np.round((self.nPatch-1)/2))
        jLeft = np.int_(jx - np.round((self.nPatch-1)/2))

        mapSlice, patchSlice = find_overlap(
            self.map,
            self.patch,
            iTop,
            jLeft
        )

        self.map[mapSlice] = np.clip(
            (self.map[mapSlice] + self.patch[patchSlice]),
            self.l_min,
            self.l_max
        )
        #'''


#-----------------------FIN NO SE USA--------------------------------------------------------------------------




def controlLoop():
   #region controlLoop setup
    global KILL_THREAD

    
    u = 0
    delta = 0
    flagtime = 0
    # used to limit data sampling to 10hz
    countMax = controllerUpdateRate / 10
    count = 0
    #endregion
    '''
    arrow1 = pg.ArrowItem(
        angle=180,
        tipAngle=60,
        headLen=10,
        tailLen=10,
        tailWidth=5,
        pen={'color': 'w', 'width': 1},
        brush='r'
    )
    arrow1.setPos(0,0)
    scope.axes[1].plot.addItem(arrow1)

    arrow2 = pg.ArrowItem(
        angle=180,
        tipAngle=60,
        headLen=10,
        tailLen=10,
        tailWidth=5,
        pen={'color': 'w', 'width': 1},
        brush='r'
    )
    scope.axes[2].plot.addItem(arrow2)
    #endregion
    '''
    #region Controller initialization
    speedController = SpeedController(
        kp=K_p,
        ki=K_i
    )
    if enableSteeringControl:
        steeringController = SteeringController(
            waypoints=waypointSequence,
            k=K_stanley,
            k_d= Kd_stanley
        )
    #endregion

    #region QCar interface setup
    qcar = QCar(readMode=1, frequency=controllerUpdateRate)
    if enableSteeringControl:
        ekf = QCarEKF(x_0=initialPose)
        gps = QCarGPS(initialPose=initialPose)
    else:
        gps = memoryview(b'') 
    #endregion

    #bsa-> Esto es para el video
    #imagencamera = qqcar.get_image(camera=3)
    #cv2.imshow("Front Camera",imagencamera[1])
    #cv2.waitKey(1)
    #cv2.destroyAllWindows()
    #------------------------

    with qcar, gps:
        t0 = time.time()
        t=0
      

        while (t < tf+startDelay) and (not KILL_THREAD):
            #region : Loop timing update
            tp = t
            t = time.time() - t0
            dt = t-tp
            #endregion

            #region : Read from sensors and update state estimates
            qcar.read()
            if enableSteeringControl:
                if gps.readGPS():
                    y_gps = np.array([
                        gps.position[0],
                        gps.position[1],
                        gps.orientation[2]
                    ])
                    ekf.update(
                        [qcar.motorTach, delta],
                        dt,
                        y_gps,
                        qcar.gyroscope[2],
                    )
                else:
                    ekf.update(
                        [qcar.motorTach, delta],
                        dt,
                        None,
                        qcar.gyroscope[2],
                    )

                x = ekf.x_hat[0,0]
                y = ekf.x_hat[1,0]
                th = ekf.x_hat[2,0]
                p = ( np.array([x, y])
                    + np.array([np.cos(th), np.sin(th)]) * 0.2)
            v = qcar.motorTach
            #endregion



            #region : Update controllers and write to car
            if t < startDelay:
                u = 0
                delta = 0
            else:
                #region BSA-> Intento desacelera
                if(steeringController.wpi >1650):
                    u = speedController.update(v, -0.4, dt)
                if(steeringController.nwp_IS_A_NODE):
                    flagtime = time.time()
                if(flagtime!=0 and time.time()-flagtime<3):
                    u = speedController.update(v, -0.4, dt)
                else:
                    u = speedController.update(v, 0.4, dt)
                    flagtime = 0
           
                        
                """
                if(flagtime is not None and time.time()-flagtime > 300):
                            u = speedController.update(v, 0.9, dt)         #u = 0
                            flagtime = None
                    
                                #time.sleep(1)
                    """
                 
                   # else:
                    #    u = speedController.update(v, v_ref, dt)

                    
                #endregion


                #region : Speed controller update

                u = speedController.update(v, v_ref, dt)
                #v_ref=0.45
                #endregion
                #print(u) 

                #region : Steering controller update
                if enableSteeringControl:
                    delta = steeringController.update(p, th, v)
                else:
                    delta = 0

                #bsa-> Este lo agregué yo
                #imagencamera = qqcar.get_image(camera=3)
                #cv2.imshow("Front Camera",imagencamera[1])
                #cv2.waitKey(1)
                #-----------
                   
                #endregion

            qcar.write(u, delta)
            #endregion

            #region : Update Scopes
            count += 1
            
            if count >= countMax and t > startDelay:
                t_plot = t - startDelay
                '''
                # Speed control scope
                speedScope.axes[0].sample(t_plot, [v, v_ref])
                speedScope.axes[1].sample(t_plot, [v_ref-v])
                speedScope.axes[2].sample(t_plot, [u])
                '''

                # Steering control scope
                if enableSteeringControl:
                    '''
                    steeringScope.axes[4].sample(t_plot, [[p[0],p[1]]])

                    p[0] = ekf.x_hat[0,0]
                    p[1] = ekf.x_hat[1,0]

                    x_ref = steeringController.p_ref[0]
                    y_ref = steeringController.p_ref[1]
                    th_ref = steeringController.th_ref

                    x_ref = gps.position[0]
                    y_ref = gps.position[1]
                    th_ref = gps.orientation[2]

                    steeringScope.axes[0].sample(t_plot, [p[0], x_ref])
                    steeringScope.axes[1].sample(t_plot, [p[1], y_ref])
                    steeringScope.axes[2].sample(t_plot, [th, th_ref])
                    steeringScope.axes[3].sample(t_plot, [delta])


                    arrow.setPos(p[0], p[1])
                    arrow.setStyle(angle=180-th*180/np.pi)
                    '''
                count = 0
                
            #endregion
            continue
        #bsa->Aquí termina la ventana
        #cv2.destroyAllWindows()  



'''

def mappingLoop():
    global KILL_THREAD, x_hat, t_hat

    og = OccupancyGrid(
        cellWidth=cellWidth,
        r_res=r_res,
        r_max=r_max,
        p_low=p_low,
        p_high=p_high
    )

    #region Configure Plots
    scope.axes[0].images[0].rotation = 90
    scope.axes[0].images[0].scale = (og.r_res, -og.phiRes*180/np.pi)
    scope.axes[0].images[0].offset = (0, 0)
    scope.axes[0].images[0].levels = (0, 1)

    scope.axes[1].images[0].scale = (og.r_res, -og.r_res)
    scope.axes[1].images[0].offset = (-og.nPatch/2, -og.nPatch/2)
    scope.axes[1].images[0].levels = (0, 1)

    scope.axes[2].images[0].scale = (og.cellWidth, -og.cellWidth)
    scope.axes[2].images[0].offset = (
        og.x_min/og.cellWidth,
        -og.y_max/og.cellWidth
    )
    scope.axes[2].images[0].levels = (0, 1)
    #endregion

    t0 = time.time()
    while time.time()-t0 < startDelay:
        gps.readLidar()

    while (not KILL_THREAD):
        #region Get latest pose estimate
        with lock:
            t = t_hat
            x = x_hat[0,0]
            y = x_hat[1,0]
            th = x_hat[2,0]

        x += 0.125 * np.cos(th)
        y += 0.125 * np.sin(th)
        #endregion

        #Read from Lidar and Update Occupancy Grid

        gps.readLidar()

        if gps.scanTime < t:
            continue

        og.updateMap(x, y, th, gps.angles,gps.distances)

        scope.axes[0].images[0].setImage(image=expit(og.polarPatch))
        scope.axes[1].images[0].setImage(image=expit(og.patch))
        scope.axes[2].images[0].setImage(image=expit(og.map))

    with lock:
        print('Mapping thread terminated')


'''


##-------------------FIN  clase carro-------------------------------------------------



# -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- --

#region : ImageInterpretation Class Setup

class ImageInterpretation():

    def __init__(self,
            imageSize,
            frameRate,
            #frameratedep,
            streamInfo,
            chessDims,
            boxSize,
          #  heigth,
            frameWidth,
            frameHeight,
            frame,
            ):

        # Camera calibration constants:
        self.NUMBER_IMAGES = 15

        # List of variables given by students
        self.imageSize      = imageSize
        self.chessboardDim  = [chessDims,chessDims]
        self.frameRate      = frameRate
        self.boxSize        = boxSize
        self.sampleRate     = 1/self.frameRate
        self.calibFinished  = False
        self.frameWidth = frameWidth
        self.frameHeigth =frameHeight
        self.frame = frame
       # self.heigth = heigth

        # List of camera intrinsic properties :
        self.CSICamIntrinsics = np.eye(3, 3, dtype=np.float32)


        [[364.98483881,   0.00,     410.42816999],
         [ 0.00,     365.86846172,    206.2004527 ],
         [ 0.,           0.00,           1.00     ]]







        # CSI camera intrinsic matrix at resolution [820, 410] is:
        # [[318.86    0.00  401.34]
        #  [  0.00  312.14  201.50]
        #  [  0.00    0.00    1.00]]
        self.CSIDistParam = np.ones((1,5),dtype= np.float32)
        # CSI camera distorion paramters at resolution [820, 410] are:

        [[-1.16319255e+00,  3.81619957e+00],
         [1.35395353e-03,  -6.03760232e-05],
         [ -5.60306025e+00]]




        # [[-0.9033  1.5314 -0.0173 0.0080 -1.1659]]

        self.d435CamIntrinsics = np.eye(3,3,dtype= np.float32)
        # D435 RGB camera intrinsic matrix at resolution [640, 480] is:

        [[482.3710721,    0.00,         317.16447774],
         [  0.00,         480.42553215, 238.8162301 ],
         [  0.00,           0.00,           1.00    ]]


        # [[455.20    0.00  308.53]
        #  [  0.00  459.43  213.56]
        #  [  0.00    0.00    1.00]]
        self.d435DistParam = np.ones((1,5), dtype= np.float32)
        # D435 RGB camera distorion paramters at resolution [640, 480] are:
        [[ 0.02553672, -0.06817711,  0.00157955, -0.00056091,  0.07795451]]





        # [[-5.1135e-01  5.4549 -2.2593e-02 -6.2131e-03 -2.0190e+01]]

        # Final Image streamed by CSI or D435 camera
        self.streamD435 = np.zeros((self.imageSize[1][0],self.imageSize[1][1]))
        self.streaCSI = np.zeros((self.imageSize[0][0],self.imageSize[0][1]))

        # Information for interfacing with front CSI camera
        enableCameras = [True, True, True, True]#PARA PODER GRUARDAR LAS IMAGENES EL FALSO SE HACE VERDADERO 
        enableCameras[streamInfo[0]] = True

        self.frontCSI = QCarCameras(
            frameWidth  = self.imageSize[0][0],
            frameHeight = self.imageSize[0][1],
            frameRate   = self.frameRate[0],
            enableRight = enableCameras[0],
            enableBack  = enableCameras[1],
            enableLeft  = enableCameras[2],
            enableFront = enableCameras[3]
        )

        # Information for interfacing with Realsense camera
        self.d435Color = QCarRealSense(
            mode=streamInfo[1],
            frameWidthRGB  = self.imageSize[1][0],
            frameHeightRGB = self.imageSize[1][1],
            frameRateRGB   = self.frameRate[1],
            frameWidthDepth=1280,
            frameHeightDepth=720,
            frameRateDepth=15,
        )

        # Initialize calibration tool:
        self.camCalibTool = ImageProcessing()

        self.SimulationTime = 70

    def camera_calibration(self):

        # saving images
        savedImages = []
        imageCount = 0
        cameraType = "csi"

        while True:
            startTime = time.time()

            # Read RGB information for front csi first, D435 rgb second
            if cameraType == "csi":
                self.frontCSI.readAll()
                endTime = time.time()
                image = self.frontCSI.csi[3].imageData
                computationTime = endTime-startTime
                sleepTime = self.sampleRate[0] \
                    - (computationTime % self.sampleRate[0])

            if cameraType == "D435":
                self.d435Color.read_RGB()
                self.d435Color.read_depth()
                endTime = time.time()
                image =self.d435Color.imageBufferRGB
                computationTime = endTime-startTime
                sleepTime = self.sampleRate[1] \
                    - (computationTime % self.sampleRate[1])

            # Use cv2 to display current image
            cv2.imshow("Camera Feed", image)

            msSleepTime = int(1000 * sleepTime)
            if  msSleepTime <= 0:
                msSleepTime = 1
            if cv2.waitKey(msSleepTime) & 0xFF == ord('q'):
                imageCount +=1
                print("saving Image #: ", imageCount)
                grayImage = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
                savedImages.append(grayImage)

                if imageCount == self.NUMBER_IMAGES and cameraType == "csi":
                    print("Implement calibration for CSI camera images: ")

                    # ===== SECTION B1 - CSI Camera Parameter Estimation =====
                    print("Camera calibration for front csi")
                    self.CSICamIntrinsics = np.eye(3, 3, dtype=np.float32)
                    self.CSIDistParam = np.ones((1, 5), dtype=np.float32)

                    # Quanser's use of OpenCV for camera calibration
                    #CALIBRACION DE LA CAMARA
                    
                    self.CSICamIntrinsics, self.CSIDistParam = \
                        self.camCalibTool.calibrate_camera(
                            savedImages,
                            self.chessboardDim,
                            self.boxSize)
                    

                    # Printed output for students
                    text = "CSI camera intrinsic matrix at resolution {} is:"
                    print(text.format(self.imageSize[0][:]))
                    print(self.CSICamIntrinsics)

                    text = ("CSI camera distortion parameters "
                        + "at resolution {} are: ")
                    print(text.format(self.imageSize[0][:]))
                    print(self.CSIDistParam)

                    cameraType = "D435"
                    savedImages = []
                    imageCount = 0

                if imageCount == self.NUMBER_IMAGES and cameraType == "D435":
                    print("Implement calibration for "
                        + "realsense D435 camera images:")

                    # ===== SECTION B2 - D435 Camera Parameter Estimation =====
                    print("Camera calibration for  D435 RGB camera")
                    self.d435CamIntrinsics = np.eye(3, 3, dtype=np.float32)
                    self.d435DistParam = np.ones((1, 5), dtype=np.float32)

                    # Quanser's use of OpenCV for camera calibration
                    
                    self.d435CamIntrinsics, self.d435DistParam = \
                        self.camCalibTool.calibrate_camera(
                            savedImages,
                            self.chessboardDim,
                            self.boxSize
                        )
                    

                    # Printed output for students
                    text = ("D435 RGB camera intrinsic matrix "
                        + "at resolution {} is:")
                    print(text.format(self.imageSize[1][:]) )
                    print(self.d435CamIntrinsics)

                    text = ("D435 RGB camera distortion parameters"
                        + "at resolution {} are: ")
                    print(text.format(self.imageSize[1][:]))
                    print(self.d435DistParam)

                    # Use completed distortion correction on D435 image
                    imageShape = np.shape(image)
                    for count, distImg in enumerate(savedImages):
                        undist = self.camCalibTool.undistort_img(
                            distImg,
                            self.d435CamIntrinsics,
                            self.d435DistParam
                        )
                        cv2.imshow("RectifiedImages", undist)
                        cv2.waitKey(500)
                    break

        print("Both Cameras calibrated!")

        self.calibFinished = True
        cv2.destroyAllWindows()
#---------------------------------------------------------------------
    def line_detection(self,cameraType):
                       #h,
                      # frameHeight=410,):
        #steering = 0

        currentTime = 0
        t0 = time.time()
        print("camera avaliable")
       

        while (currentTime < self.SimulationTime):
            LoopStartTime = time.time()
            currentTime = time.time()-t0

            # Check which stream will be used for line detection:
            if cameraType == "csi":

                self.frontCSI.readAll()
                endTime = time.time()
                image = self.frontCSI.csi[3].imageData
                fr = self.frame
                computationTime = endTime-LoopStartTime
                sleepTime = self.sampleRate[0] \
                    - (computationTime % self.sampleRate[0])
                cameraIntrinsics = self.CSICamIntrinsics
                cameraDistortion = self.CSIDistParam

            if cameraType == "D435":

                self.d435Color.read_RGB()
                endTime = time.time()
                image = self.d435Color.imageBufferRGB
                image2 = self.d435Color.imageBufferRGB
               # frame = self.d435Color.frameHeightRGB.frame
                frameh = self.frameHeigth
                fr = self.frame
                computationTime = endTime-LoopStartTime
                sleepTime = self.sampleRate[1] \
                    - (computationTime % self.sampleRate[1])
                cameraIntrinsics = self.d435CamIntrinsics
                cameraDistortion = self.d435DistParam

      
#----------------------------------COMIENZA LA PRUEBA--------------------------------------------
      
                #----------------------------INICIO IF PRUEBA------------------------------------
            '''
                    frameWidthRGB=1920,
                    frameHeightRGB=1080,


                    imageSize=[[820,410], [640,480]],

            

                    
           
            # ============= SECTION C1 - Image Correction =============
            print("Implement image correction for raw camera image... ")
            undistortedImage = image


            # Quanser's use of OpenCV for distortion correction
            
            imageShape = np.shape(image)
            undistortedImage = self.camCalibTool.undistort_img(
                image,
                cameraIntrinsics,
                cameraDistortion
            )
            

            # ============= SECTION C2 - Image Filtering =============
            print("Implement image filter on distortion corrected image... ")
            filteredImage = image

            # Quanser's solution for image filtering for feature detection
            
            filteredImage = self.camCalibTool.do_canny(undistortedImage)
            # filteredImage = self.camCalibTool.doSobel(image)
            

            # ============= SECTION C3 - Feature Extraction =============
            print("Extract line information from filtered image... ")
            linesImage, lines = image, []

            # Quanser's solution for feature extraction
            
            linesImage, lines = self.camCalibTool.extract_lines(
                filteredImage,
                undistortedImage
            )

          
            print("Dosegment... ")
           
            segment,mask  = image,[]
                      
            # Quanser's solution for feature extraction
            #imageShape = np.shape(image)
            #Image = np.copy(image)
            segment, mask = self.camCalibTool.do_segment(undistortedImage,steering=0
            )

           


            
            print("Dosegment... ")

           # grayS = cv2.cvtColor(segment, cv2.COLOR_RGB2GRAY)

            rigth,left = self.camCalibTool.calculate_lines(filteredImage)



            #print(rigth)
            #print(left)
            #print(lines)
                        
            imageDisplayed = image
            imageDisplayed1 = filteredImage
            imageDisplayed2 = linesImage
            imageDisplayed3 = segment
            #imageDisplayed4 = vizu
            #imageDisplayed5 = mask

            # Use cv2 to display current image
            #cv2.imshow("Lines Image", imageDisplayed)  
            #cv2.imshow("FILTRO", imageDisplayed1)
            #cv2.imshow("LINES", imageDisplayed2)  
            cv2.imshow("SEGMENT", imageDisplayed3)  
            #cv2.imshow("LINES", imageDisplayed4)  
            #cv2.imshow("SEGMENT", imageDisplayed5)

            msSleepTime = int(1000*sleepTime)
            if  msSleepTime >= 0:
                msSleepTime = 1

            cv2.waitKey(msSleepTime)

            '''

               # ============= SECTION C1 - Image Correction =============
            #print("Implement image correction for raw camera image... ")
            undistortedImage = image


            # Quanser's use of OpenCV for distortion correction

            #----------IMAGEN BASE PARA NO TENER DISTORSION------USAR DE BASE EN LUGAR DE "IMAGE"-------
            
            imageShape = np.shape(image)
            undistortedImage = self.camCalibTool.undistort_img(
                image,
                cameraIntrinsics,
                cameraDistortion
            )

            #----FIN IMAGEN BASE "UNDISTOREDIMAGE"------------------------------------
            
            
            rojo=np.copy(undistortedImage)
            rojohsv = cv2.cvtColor(rojo, cv2.COLOR_BGR2HSV)
            stop= self.camCalibTool.detect_red_laneBAJO(rojohsv)


            rojo2=np.copy(undistortedImage)
            rojohsv2 = cv2.cvtColor(rojo2, cv2.COLOR_BGR2HSV)
            stop2= self.camCalibTool.detect_red2_lane2ALTO(rojohsv2)

            RojoD = cv2.add(stop,stop2)
    
            mask = np.copy(RojoD)
            BinariChida = image
            BinariChida = self.camCalibTool.image_filtering_open(mask)#MASCARA FILTRADA PARA UN MEJOR RESULADO

            masktwo = np.copy(RojoD)
            BinariChidatwo = image
            BinariChidatwo = self.camCalibTool.image_filtering_close(masktwo)#MASCARA FILTRADA PARA UN MEJOR RESULADO


            #BinRGB = cv2.cvtColor(BinariChida,cv2.COLOR_HSV2RGB)
            BinDef = cv2.add(BinariChida,BinariChidatwo)

            cc = np.copy(BinDef)
            liness, line = image, []
            liness, line = self.camCalibTool.extract_lines(cc,undistortedImage)


            BinDefCopy = np.copy(BinDef)
           # MaskFramed = image
           # MaskFramed = self.camCalibTool.mask_image(BinDefCopy,20,240,310,610
            Row = self.camCalibTool.extract_lane_points_by_row(BinDefCopy,160)
            #print (Row)

            (r1,r2),(c1,c2) = Row      
            lines = np.copy(liness)


            self.camCalibTool.circle_pts(lines,Row, 30,(255,0,0))

                 #frameundi = np.copy(undistortedImage)
            #MaskDef = np.copy(BinDef)
            #segment, mask = image,[]
            #segment, mask = self.camCalibTool.do_segment(undistortedImage, steering = 0)


            #self.camCalibTool.circle_pts(lines,Row, 50,(255,0,0))
            frameundi = np.copy(undistortedImage)

            
            MaskSem = np.copy(BinDef)
            frd = image
            frd = self.camCalibTool.mask_image(MaskSem,(50),(250),(280),(340))

            MaskDef = np.copy(BinDef)
            contornos,jerarquia = cv2.findContours(MaskDef,cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)
            cv2.drawContours(frameundi,contornos,-1,(0,255,0),2)

            MaskSemvrgs = np.copy(BinDef)
            MaskVRGS = image
            MaskVRGS = cv2.add(MaskSemvrgs,frd)



            '''
            for i, contorno in enumerate(contornos):
                if i == 0:
                    continue

                epsilon = 0.01*cv2.arcLength(contorno, True)
                approx = cv2.approxPolyDP(contorno, epsilon, True)

                x, y, w, h= cv2.boundingRect(approx)
                x_mid = int(x + (w/2))
                y_mid = int(y + (h/2))

                coords = (x_mid, y_mid)
                colour = (0, 0, 0)
                font = cv2.FONT_HERSHEY_DUPLEX

                if len(approx) == 8:
                    cv2.putText(frameundi, "STOP", coords, font, 1, colour, 1)
                elif len(approx) > 8:
                    cv2.putText(frameundi, "stoplight", coords, font, 1, colour, 1)

                '''
            
            hsvImage = cv2.cvtColor(frameundi, cv2.COLOR_BGR2HSV)
            yellow_mask = cv2.inRange(hsvImage, lowerBounds_amarillo, upperBounds_amarillo)
            stoplight_feed = cv2.bitwise_and(frameundi, frameundi, mask=yellow_mask)
            hsvImage = cv2.cvtColor(stoplight_feed, cv2.COLOR_BGR2HSV)

            redlight_mask = cv2.inRange(hsvImage, lowerBounds_lightRed, upperBounds_lightRed)

            stoplight_contours, _ = cv2.findContours(redlight_mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
            if stoplight_contours is not None:
                for i, contour in enumerate(stoplight_contours):
                    if i == 0:
                        continue

            #epsilon = 0.05*cv2.arcLength(contour, True)
            #approx = cv2.approxPolyDP(contour, epsilon, True)

                    x, y, w, h= cv2.boundingRect(contour)
                    x_mid = int(x + (w/2))
                    y_mid = int(y + (h/2))

                    coords = (x_mid, y_mid)
                    colour = (0, 0, 0)
                    font = cv2.FONT_HERSHEY_DUPLEX
                    cv2.putText(frameundi, "SEMAFORO ROJO PUTO", coords, font, 1, colour, 1)
            ##VELOCIDAD MOTOR-------------------------
            #MaskMot = np.copy(BinDef)


            #slop, intercept = self.camCalibTool.find_slope_intercept_from_binary(MaskMot)

            #print(slop)
            #print(intercept)

            #coordenadas = self.camCalibTool.calculate_coordinates((slop,intercept))

            #print(coordenadas)
            #x1,y1,x2,y2 = coordenadas

            #lineparameters = (x1, x2), (y1, y2) # coordenadas
            #print(lineparameters)

         
            #r,l = self.camCalibTool.calculate_lines(liness)
            #print(r)
            #print(l)
     
        
            #vel,steer = self.camCalibTool.driving_parameters(lineparameters)
             #def calculate_coordinates(self, parameters):
            #print(vel)
            #print(steer)

            #print(l)
            



            #-----------------------
   
            #print("Display image with lines found... ")
      
            
            #imageDisplayed8 = circulo
          
            imageDisplayed12 = lines
            imageDisplayed13 = BinDef
            imageDisplayed14 = frameundi
            imageDisplayed15 = frd
            imageDisplayed16 = MaskVRGS




            # Use cv2 to display current image
            #cv2.imshow("BORDES CON CANNY", imageDisplayed)  #-----------------LISTA
           # cv2.imshow("CANNY CON FILTRO", imageDisplayed0)#-----------------------READY
            #cv2.imshow("FILTRO", imageDisplayed1)#-----------------------READY
            #cv2.imshow("S", imageDisplayed2)  $------------------READY
            #cv2.imshow("lineas AZULES", imageDisplayed3)  #----------READY
            #cv2.imshow("lineas AZULES filtro", imageDisplayed31)  #----------READY
            #cv2.imshow("BINARIA", imageDisplayed4)  
            #cv2.imshow("CLEAN IMAGE CANNY", imageDisplayed5)
            #cv2.imshow("ESCALA DE GRISES", imageDisplayed6)#--------------------LISTA
            #cv2.imshow("CANNY", imageDisplayed7)#----------------READY
            #cv2.imshow("CIRCULO", imageDisplayed8)

           
            #cv2.imshow("Lineas en MASCARA", imageDisplayed12)
            cv2.imshow("MASK", imageDisplayed13)
            cv2.imshow("STOP DETECTION", imageDisplayed14)
            #cv2.imshow(" DEFINITIVA", imageDisplayed15)
            #cv2.imshow(" DETIVA", imageDisplayed16)
       
       
       

           
          

            msSleepTime = int(1000*sleepTime)
            if  msSleepTime >= 0:
                msSleepTime = 1

            cv2.waitKey(msSleepTime)


    def stop_cameras(self):
        # Stopping the image feed for both cameras
        self.frontCSI.terminate()
        self.d435Color.terminate()

#endregion

# -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- --

#region : Main

#-------------------------------------------------------------------LISTO------------------------
def main():
    
    try:
        '''
        INPUTS:
        imageSize           = [[L,W],[L,W]] 2x2 array which specifies the
                                resultion for the CSI camera and the D435
        frameRate           = [CSIframeRate, D435frameRate] 2x1 array for
                                image frame rates for CSI, D435 camera
        streamInfo          = [CSIIndex,D435Stream] 2x1 array to specify the
                                CSI camera index and the D435 image stream
        chessDims           = Specify the number of cells of the square
                                chessboard used for calibration
        boxSize             = Float value to specify the size of the grids
                                in the chessboard. Be mindful of the physical
                                units being used
        '''

        # ======== SECTION A - Student Inputs for Image Interpretation ===========
        cameraInterfacingLab = ImageInterpretation(
            imageSize=[[820,410], [640,480]],
            frameRate=np.array([30, 30]),
            streamInfo=[3, "RGB"],
            chessDims=6,
            boxSize=1,
            frameWidth=820,
            frameHeight=410,
            frame= 900
           # heigth= [1][0]
        )

        ''' Students decide the activity they would like to do in the
        ImageInterpretation Lab

        List of current activities:
        - Calibrate   (interfacing skill activity)
        - Line Detect (line detection skill activity)
        '''
        camMode = "Line Detect"

        # ========= SECTION D - Camera Intrinsics and Distortion Coeffs. =========
        cameraMatrixCSI  = np.array([[364.98483881,   0.,         410.42816999],
                                  [  0.00,     365.86846172,   206.2004527 ],
                                  [  0.00,           0.00,       1.        ]])
        
        distortionCoefficientsCSI = np.array([
           -1.16319255e+00,
           3.81619957e+00, 
           1.35395353e-03,
           -6.03760232e-05,
           -5.60306025e+00,
        ])


        cameraMatrixD435  = np.array(  [[482.3710721,    0.00,         317.16447774],
                                     [  0.00,         480.42553215, 238.8162301 ],
                                     [  0.00,           0.00,           1.00    ]])
        
        distortionCoefficientsD435 = np.array([
          0.02553672, 
          -0.06817711,  
          0.00157955, 
          -0.00056091,  
          0.07795451
        ])
        
        if camMode == "Calibrate":
            try:
                cameraInterfacingLab.camera_calibration()
                if cameraInterfacingLab.calibFinished == True \
                        and camMode == "Calibrate":
                    print("calibration process done, stopping cameras...")
                    cameraInterfacingLab.stop_cameras()

            except KeyboardInterrupt:
                cameraInterfacingLab.stop_cameras()

        if camMode == "Line Detect":
            try:
                text = "Specify the camera used for line detection (csi/D435): "
                cameraType = input(text)
                if cameraType == "csi" :
                    #while controlThread.is_alive() and mappingThread.is_alive():
                    cameraInterfacingLab.CSICamIntrinsics = cameraMatrixCSI
                    cameraInterfacingLab.CSIDistParam     = distortionCoefficientsCSI
                    cameraInterfacingLab.line_detection(cameraType)
                  
                
                elif cameraType =="D435":
                    #----inicio


                    
                    #controlThread.start()
                    #mappingThread.start()
                    #MultiScope.refreshAll()
                    #time.sleep(0.01)
                    loop.start()
                    #controlLoop()
                    cameraInterfacingLab.d435CamIntrinsics = cameraMatrixD435
                    cameraInterfacingLab.d435DistParam     = distortionCoefficientsD435
                    cameraInterfacingLab.line_detection(cameraType)
                    #while controlThread.is_alive() and mappingThread.is_alive():
                    
                
                    #MultiScope.refreshAll()
                    #time.sleep(0.01)
                   

                    
                
#-------------------------------------------LINEA AGREGADA---------------------------------------------------------

            

#----------------#-------------------------FIN DEL ELIF HECHO -------------------------------------------------
                else:
                    print("Invalid camera type")

            except KeyboardInterrupt:
                cameraInterfacingLab.stop_cameras()
    finally:

        if not IS_PHYSICAL_QCAR:
            import Setup_Competition
            Setup_Competition.terminate()
        
            #gps.terminate()
        input('Experiment complete. Press any key to exit...')


#endregion



# -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- --

#----------------------------SPEED.SCOPE--------------------------------------------------------


#------------------
'''
    #region : Setup scopes
if IS_PHYSICAL_QCAR:
    fps = 10
else:
    fps = 30
    # Scope for monitoring speed controller
    speedScope = MultiScope(
        rows=3,
        cols=1,
        title='Vehicle Speed Control',
        fps=fps
    )
    speedScope.addAxis(
        row=0,
        col=0,
        timeWindow=tf,
        yLabel='Vehicle Speed [m/s]',
        yLim=(0, 1)
    )
    speedScope.axes[0].attachSignal(name='v_meas', width=2)
    speedScope.axes[0].attachSignal(name='v_ref')

    speedScope.addAxis(
        row=1,
        col=0,
        timeWindow=tf,
        yLabel='Speed Error [m/s]',
        yLim=(-0.5, 0.5)
    )
    speedScope.axes[1].attachSignal()

    speedScope.addAxis(
        row=2,
        col=0,
        timeWindow=tf,
        xLabel='Time [s]',
        yLabel='Throttle Command [%]',
        yLim=(-0.3, 0.3)
    )
    speedScope.axes[2].attachSignal()

    # Scope for monitoring steering controller
    #if enableSteeringControl:
    steeringScope = MultiScope(
            rows=4,
            cols=2,
            title='Vehicle Steering Control',
            fps=fps
        )

    steeringScope.addAxis(
            row=0,
            col=0,
            timeWindow=tf,
            yLabel='x Position [m]',
            yLim=(-2.5, 2.5)
        )
    steeringScope.axes[0].attachSignal(name='x_meas')
    steeringScope.axes[0].attachSignal(name='x_ref')

    steeringScope.addAxis(
            row=1,
            col=0,
            timeWindow=tf,
            yLabel='y Position [m]',
            yLim=(-1, 5)
        )
    steeringScope.axes[1].attachSignal(name='y_meas')
    steeringScope.axes[1].attachSignal(name='y_ref')

    steeringScope.addAxis(
            row=2,
            col=0,
            timeWindow=tf,
            yLabel='Heading Angle [rad]',
            yLim=(-3.5, 3.5)
        )
    steeringScope.axes[2].attachSignal(name='th_meas')
    steeringScope.axes[2].attachSignal(name='th_ref')

    steeringScope.addAxis(
            row=3,
            col=0,
            timeWindow=tf,
            yLabel='Steering Angle [rad]',
            yLim=(-0.6, 0.6)
        )
    steeringScope.axes[3].attachSignal()
    steeringScope.axes[3].xLabel = 'Time [s]'

    steeringScope.addXYAxis(
            row=0,
            col=1,
            rowSpan=4,
            xLabel='x Position [m]',
            yLabel='y Position [m]',
            xLim=(-2.5, 2.5),
            yLim=(-1, 5)
        )

    im = cv2.imread(
            images.SDCS_CITYSCAPE,
            cv2.IMREAD_GRAYSCALE
        )

    steeringScope.axes[4].attachImage(
            scale=(-0.002035, 0.002035),
            offset=(1125,2365),
            rotation=180,
            levels=(0, 255)
        )
    steeringScope.axes[4].images[0].setImage(image=im)

    referencePath = pg.PlotDataItem(
            pen={'color': (85,168,104), 'width': 2},
            name='Reference'
        )
    steeringScope.axes[4].plot.addItem(referencePath)
    referencePath.setData(waypointSequence[0, :],waypointSequence[1, :])

    steeringScope.axes[4].attachSignal(name='Estimated', width=2)

    arrow = pg.ArrowItem(
            angle=180,
            tipAngle=60,
            headLen=10,
            tailLen=10,
            tailWidth=5,
            pen={'color': 'w', 'fillColor': [196,78,82], 'width': 1},
            brush=[196,78,82]
        )
    arrow.setPos(initialPose[0], initialPose[1])
    steeringScope.axes[4].plot.addItem(arrow)
    #endregion


'''


'''
scope = MultiScope(
            rows=2,
            cols=2,
            title='Environment Interpretation',
            fps=fps
        )

        # Polar Patch
scope.addXYAxis(
            row=0,
            col=0,
            xLabel='Angle [deg]',
            yLabel='Range [m]'
        )
scope.axes[0].attachImage()

        # Patch
scope.addXYAxis(
            row=1,
            col=0,
            xLabel='x Position [m]',
            yLabel='y Position [m]'
        )
scope.axes[1].attachImage()

        # Generated Map and followed trajectory
scope.addXYAxis(
            row=0,
            col=1,
            rowSpan=2,
            xLabel='x Position [m]',
            yLabel='y Position [m]',
            xLim=(-4, 3),
            yLim=(-2, 6)
        )
scope.axes[2].attachSignal(name='Measured', width=2, style='--.')
scope.axes[2].attachImage()

referencePath = pg.PlotDataItem(
            pen={'color': (85,168,104), 'width': 2},
            name='Reference'
        )
referencePath.setData(waypointSequence[0, :], waypointSequence[1, :])
scope.axes[2].plot.addItem(referencePath)



        #endregion

    #region : Setup threads, then run experiment

#mappingThread = Thread(target=mappingLoop)

'''
'''

#mappingThread.start()


'''

#------------------SCOPE INITIAL------------------------------------------------------------

#-------------------IGNORAR-------------------------------------------------------------
#region : Run
if __name__ == '__main__':
        #region : Setup scopes
 
    #region : Setup control thread, then run experiment


    
    loop = Thread(target=controlLoop)
    controlThread = Thread(target=main)
    controlThread.start()
    #main() 

    try:
        while controlThread.is_alive() and (not KILL_THREAD):
            
            #controlLoop()
            #MultiScope.refreshAll()
            time.sleep(0.01)

    finally:
        KILL_THREAD = True
    #endregion
    if not IS_PHYSICAL_QCAR:
        Setup_Competition.terminate()



    '''
    
    controlThread = Thread2(target=controlLoop)
    #mainThread = Thread(target=main)

    #if camMode == "Line Detect":



    mainThread.start()
    try:
        while mainThread.is_alive() and (not KILL_THREAD):

            controlThread.start()
            controlLoop()
            MultiScope.refreshAll()
            time.sleep(0.01)


        #mainThread.start()



    finally:
        KILL_THREAD = True
    #endregion
    if not IS_PHYSICAL_QCAR:
        Setup_Competition.terminate()
    '''

    '''


    try:
        while controlThread.is_alive() and (not KILL_THREAD):
            MultiScope.refreshAll()
            time.sleep(0.01)
    finally:
        KILL_THREAD = True
    #endregion
    if not IS_PHYSICAL_QCAR:
        Setup_Competition.terminate()





    controlThread.start()
    #controlLoop()
    #main()

 
    input('Experiment complete. Press any key to exit...')

    
    '''
   

  
    
    #region : Setup Scopes

#endregion
   
#endregion