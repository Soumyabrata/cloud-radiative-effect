import numpy as np

OCAM_MODEL_WAHRSIS1 = {'ss': np.array([-1000.420204765826, 0, 0.0004143028616796895, -0.0000001326602517023016, 0.0000000001189063438897328]),
              'xc': 1699.292086924051,
              'yc': 2623.857945863851,
              'c': 1.001497578774721,
              'd': 0.0001648328572544254,
              'e': -0.0003701714803382626,
              'width': 5184,
              'height': 3456,
              'pol': np.array([17.093112021290, 116.906832105765, 302.590771505238, 358.151930866653, 198.778850887297, 122.119010267068, 142.058271377689, -29.881556376064, 796.234464713210, 1454.477993845334])
          }

OCAM_MODEL_WAHRSIS3 = {'ss': np.array([-940.0536263327799, 0, -0.0004350078189360874, 0.000001992235625021194, -0.000000001852994427341681, 0.0000000000006287859561021287]),
              'xc': 1726.489990802254,
              'yc': 2595.583938922679,
              'c': 1,
              'd': 0,
              'e': 0,
              'width': 5184,
              'height': 3456,
              'pol': np.array([-1166.985385855834, -12615.37445538732, -60167.43456926524, -166119.1504969074, -292426.6785305706, -340068.9318239510, -261283.4698211940, -128980.5723102148, -39142.61635247536, -7226.750896152065, -568.4425015568790, 247.6483120616119, -285.3734583902227, 587.9778246127931, 1426.899976068007])
          }

OCAM_MODEL_RESIZED = {'ss': np.array([-196.0652346266524, 0, 0.001551354130936, 0.000002271715547157454]),
              'xc': 333,
              'yc': 500,
              'c': 1.001518936559905,
              'd': -0.0006074390779041386,
              'e': -0.0006157623771845934,
              'width': 1000,
              'height': 667,
              'pol': np.array([0.5824584941954, 2.7882658700455, 4.5626523182331, 4.3499398597064, 7.2151606816907, 11.7412839474610, 21.5284591696463, 44.0268208367333, 38.7907086632400, 195.0923410592154, 296.8080381628882])
          }
                      

# Project a give pixel point onto the unit sphere
#   M=CAM2WORLD=(m, ocam_model) returns the 3D coordinates of the vector
#   emanating from the single effective viewpoint on the unit sphere
def cam2world(m, ocam_model = None):
    
    if ocam_model:
        n_points = m.shape[1]
        ss = ocam_model['ss']
        xc = ocam_model['xc']
        yc = ocam_model['yc']
        c = ocam_model['c']
        d = ocam_model['d']
        e = ocam_model['e']

        A = np.array([[c, d], [e, 1]])
        T = np.tile(np.array([[xc], [yc]]), (1, n_points))

        m = np.dot(np.linalg.inv(A), m-T)

        M = np.array([m[0,:], m[1,:], np.polyval(ss[ ::-1], np.sqrt(np.power(m[0,:], 2) + np.power(m[1,:], 2)))])
        M = M/np.tile(np.sqrt(np.power(M[0,:], 2) + np.power(M[1,:], 2) +  + np.power(M[2,:], 2)), (3, 1))
        return M
        
    else:
        m1 = m[0,:] - 1728
        m2 = m[1,:] - 2592
        azimuths = np.arctan2(m2, m1)

        r = np.sqrt(np.power(m1,2) + np.power(m2,2))
        theta = np.pi/2 - 2*np.arcsin(r/(1472/np.sin(np.pi/4)))

        M1 = m1
        M2 = m2
        M3 = r*np.tan(-theta)
        norm = np.sqrt(np.power(M1, 2) + np.power(M2, 2) + np.power(M3, 2))

        return np.column_stack((M1/norm, M2/norm, M3/norm)).T



# Projects a 3D point on to the image
#   m=WORLD2CAM_FAST(M, ocam_model) projects a 3D point on to the
#   image and returns the pixel coordinates. This function uses an approximation of the inverse
#   polynomial to compute the reprojected point. Therefore it is very fast.
def world2cam(M, ocam_model = None):
    
    if ocam_model:
        xc = ocam_model['xc']
        yc = ocam_model['yc']
        c = ocam_model['c']
        d = ocam_model['d']
        e = ocam_model['e']
        pol = ocam_model['pol']

        norm = np.sqrt(np.power(M[0,:], 2) + np.power(M[1,:], 2))
        norm[norm == 0] = np.spacing(1)

        theta = np.arctan(M[2,:] / norm)
        rho = np.polyval(pol, theta)

        x = M[0,:]/norm*rho
        y = M[1,:]/norm*rho

        m = np.array([x*c + y*d + xc, x*e + y + yc])
        return m
    else:
        azimuths = np.arctan2(M[0,:], M[1,:])
        elevations = np.arccos(-M[2,:]/np.sqrt(np.power(M[0,:],2) + np.power(M[1,:],2) + np.power(M[2,:],2)))

        r = 1472/np.sin(np.pi/4)*np.sin(elevations/2)
        m1 = 1728 + r*np.sin(azimuths)
        m2 = 2592 + r*np.cos(azimuths)

        return np.column_stack((m1, m2)).T
