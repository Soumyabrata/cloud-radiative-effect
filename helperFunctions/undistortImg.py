import numpy as np
from scipy import interpolate
import internal_calibration
#import pycuda.autoinit
#import pycuda.driver as drv
#from pycuda.compiler import SourceModule

def undistortCC(img, sizeResult = [500, 500, 3] , altitude = 150 , rot = np.array([[1, 0, 0], [0, 1, 0], [0, 0, 1]]) , trans = np.array([[0], [0], [0]]) , bearing = 0 , elevation = 0 , pixelWidth=1, cuda=False):
    """Undistort a fish-eye image to a standard camera image by a ray-tracing approach.
    img: the fish-eye image to undistort
    sizeResult: the size of the output undistorted image [pixels]
    altitude: the altitude of the virtual image plane to form the undistorted image [meters]
    rot: a rotation matrix (3x3) to cope with the misalignement of the image
    trans: a translation matrix (3x1) to cope with the misalignement of the image
    bearing: the bearing angle of the undistorted plane (in degrees)
    elevation: the elevation angle of the undistorted plane (in degrees)
    pixelWidth: the width in real world coordinates of one pixel of the undistorted image [meters]
    cuda: set to True if cuda (GPU) interpolation is used
    """

    centerImage = np.array([sizeResult[1]/2, sizeResult[0]/2])
    
    if len(np.shape(img)) == 3:
        R = img[:,:,0]
        G = img[:,:,1]
        B = img[:,:,2]
    
    sizeImg = img.shape
    posX, posY = np.meshgrid(np.arange(sizeImg[1]), np.arange(sizeImg[0]))
    posX = posX + 1
    posY = posY + 1

    mapY, mapX = np.meshgrid(np.arange(sizeResult[1]), np.arange(sizeResult[0]))
    mapX = mapX + 1
    mapY = mapY + 1
    distCenterX = (mapX.ravel(order='F') - centerImage[0]) * pixelWidth
    distCenterY = (mapY.ravel(order='F') - centerImage[1]) * pixelWidth
    distCenterZ = np.tile(-altitude, (distCenterX.shape[0], 1))
    
    bearing = np.radians(bearing)
    elevation = np.radians(elevation)
    rotHg = np.array([[np.cos(elevation), 0, np.sin(elevation)], [0, 1, 0], [-np.sin(elevation), 0, np.cos(elevation)]])
    rotBearing = np.array([[np.cos(bearing), -np.sin(bearing), 0], [np.sin(bearing), np.cos(bearing), 0], [0, 0, 1]])
    
    worldCoords = np.column_stack((distCenterX, distCenterY, distCenterZ)).T
    
    worldCoords = np.dot(rotHg, worldCoords)
    worldCoords = np.dot(rotBearing, worldCoords)
    worldCoords = np.dot(rot.T, worldCoords - trans)
    
    m = internal_calibration.world2cam(worldCoords)

    if len(np.shape(img)) == 3:
        if cuda:
            result = cuda_interpolate3D(img, m, (sizeResult[0], sizeResult[1]))
        else:
            ip = interpolate.RectBivariateSpline(np.arange(sizeImg[0]), np.arange(sizeImg[1]), R)
            resultR = ip.ev(m[0,:]-1, m[1,:]-1)
            resultR = resultR.reshape(sizeResult[0], sizeResult[1],order='F')
            np.clip(resultR, 0, 255, out=resultR)
            resultR = resultR.astype('uint8')

            ip = interpolate.RectBivariateSpline(np.arange(sizeImg[0]), np.arange(sizeImg[1]), G)
            resultG = ip.ev(m[0,:]-1, m[1,:]-1)
            resultG = resultG.reshape(sizeResult[0], sizeResult[1],order='F')
            np.clip(resultG, 0, 255, out=resultG)
            resultG = resultG.astype('uint8')

            ip = interpolate.RectBivariateSpline(np.arange(sizeImg[0]), np.arange(sizeImg[1]), B)
            resultB = ip.ev(m[0,:]-1, m[1,:]-1)
            resultB = resultB.reshape(sizeResult[0], sizeResult[1],order='F')
            np.clip(resultB, 0, 255, out=resultB)
            resultB = resultB.astype('uint8')

            result = np.zeros(sizeResult).astype('uint8')
            result[:,:,0] = resultR
            result[:,:,1] = resultG
            result[:,:,2] = resultB
    else:
        if cuda:
            result = cuda_interpolate(img, m, (sizeResult[0], sizeResult[1]))
        else:
            ip = interpolate.RectBivariateSpline(np.arange(sizeImg[0]), np.arange(sizeImg[1]), img)
            result = ip.ev(m[0,:]-1, m[1,:]-1)
            result = result.reshape(sizeResult[0], sizeResult[1],order='F')
            np.clip(result, 0, 255, out=result)
            result = result.astype('uint8')
    
    return result


def cuda_interpolate(channel, m, size_result):
    cols = size_result[0]; rows = size_result[1];

    kernel_code = """
    texture<float, 2> tex;

    __global__ void interpolation(float *dest, float *m0, float *m1)
    {
      int idx = threadIdx.x + blockDim.x * blockIdx.x;
      int idy = threadIdx.y + blockDim.y * blockIdx.y;

      if (( idx < %(NCOLS)s ) && ( idy < %(NDIM)s )) {
        dest[%(NDIM)s * idx + idy] = tex2D(tex, m0[%(NDIM)s * idy + idx], m1[%(NDIM)s * idy + idx]);
      }
    }
    """
    
    kernel_code = kernel_code % {'NCOLS': cols, 'NDIM': rows}
    mod = SourceModule(kernel_code)

    interpolation = mod.get_function("interpolation")
    texref = mod.get_texref("tex")

    channel = channel.astype("float32")
    drv.matrix_to_texref(channel, texref, order="F")
    texref.set_filter_mode(drv.filter_mode.LINEAR)

    bdim = (16, 16, 1)
    dx, mx = divmod(cols, bdim[0])
    dy, my = divmod(rows, bdim[1])

    gdim = ((dx + (mx>0)) * bdim[0], (dy + (my>0)) * bdim[1])

    dest = np.zeros((rows,cols)).astype("float32")
    m0 = (m[0,:]-1).astype("float32")
    m1 = (m[1,:]-1).astype("float32")

    interpolation(drv.Out(dest), drv.In(m0), drv.In(m1), block=bdim, grid=gdim, texrefs=[texref])

    return dest.astype("uint8")


def cuda_interpolate3D(img, m, size_result):
    cols = size_result[0]; rows = size_result[1];

    kernel_code = """
    texture<float, 2> texR;
    texture<float, 2> texG;
    texture<float, 2> texB;

    __global__ void interpolation(float *dest, float *m0, float *m1)
    {
      int idx = threadIdx.x + blockDim.x * blockIdx.x;
      int idy = threadIdx.y + blockDim.y * blockIdx.y;

      if (( idx < %(NCOLS)s ) && ( idy < %(NDIM)s )) {
        dest[3*(%(NDIM)s * idx + idy)] = tex2D(texR, m0[%(NDIM)s * idy + idx], m1[%(NDIM)s * idy + idx]);
        dest[3*(%(NDIM)s * idx + idy) + 1] = tex2D(texG, m0[%(NDIM)s * idy + idx], m1[%(NDIM)s * idy + idx]);
        dest[3*(%(NDIM)s * idx + idy) + 2] = tex2D(texB, m0[%(NDIM)s * idy + idx], m1[%(NDIM)s * idy + idx]);
      }
    }
    """
    
    kernel_code = kernel_code % {'NCOLS': cols, 'NDIM': rows}
    mod = SourceModule(kernel_code)

    interpolation = mod.get_function("interpolation")
    texrefR = mod.get_texref("texR")
    texrefG = mod.get_texref("texG")
    texrefB = mod.get_texref("texB")

    img = img.astype("float32")
    drv.matrix_to_texref(img[:,:,0], texrefR, order="F")
    texrefR.set_filter_mode(drv.filter_mode.LINEAR)
    drv.matrix_to_texref(img[:,:,1], texrefG, order="F")
    texrefG.set_filter_mode(drv.filter_mode.LINEAR)
    drv.matrix_to_texref(img[:,:,2], texrefB, order="F")
    texrefB.set_filter_mode(drv.filter_mode.LINEAR)

    bdim = (16, 16, 1)
    dx, mx = divmod(cols, bdim[0])
    dy, my = divmod(rows, bdim[1])

    gdim = ((dx + (mx>0)) * bdim[0], (dy + (my>0)) * bdim[1])

    dest = np.zeros((rows,cols,3)).astype("float32")
    m0 = (m[0,:]-1).astype("float32")
    m1 = (m[1,:]-1).astype("float32")

    interpolation(drv.Out(dest), drv.In(m0), drv.In(m1), block=bdim, grid=gdim, texrefs=[texrefR, texrefG, texrefB])

    return dest.astype("uint8")
