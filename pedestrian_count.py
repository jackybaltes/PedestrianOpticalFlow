	

    #!/usr/bin/env python2
     
    import numpy as np
    import cv2
    import math
    import linecache
    import sys
    import matplotlib.pyplot as plt
     
    help_message = '''
    USAGE: pedestrian_count.py [<video_source>+]
    '''
     
    def draw_flow(img, flow, step=4):
        h, w = img.shape[:2]
        y, x = np.mgrid[step/2:h:step, step/2:w:step].reshape(2,-1)
        fx, fy = flow[y,x].T
        lines = np.vstack([x, y, x+fx, y+fy]).T.reshape(-1, 2, 2)
        lines = np.int32(lines + 0.5)
        vis = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
        cv2.polylines(vis, lines, 0, (0, 255, 0))
        #for (x1, y1), (x2, y2) in lines:
        #    cv2.circle(vis, (x1, y1), 1, (0, 255, 0), -1)
        return vis
     
    def UpdateBuckets( flow, buckets, thres ):
        h, w = flow.shape[:2]
        fx, fy = flow[:,:,0], flow[:,:,1]
        ang = np.arctan2(fy, fx) + np.pi
        v = np.sqrt(fx*fx+fy*fy)
       
        tv = v > thres
    #    print tv
        for i in range( len( buckets ) ):
            baLow = ang >= ( ( 2.0 * math.pi ) / len(buckets) * i ) - (2 * math.pi / (2*len(buckets ) ) )
            baHigh = ang < ( ( 2.0 * math.pi ) / len(buckets) * (i + 1 ) - ( 2 * math.pi / (2*len(buckets) ) ) )
            ba = baLow * baHigh
            bb = ( tv * v ) * ba
            cnt = sum( bb.reshape(-1) )
            buckets[i] = buckets[i] + cnt
                   
    def draw_hsv(flow, thres):
        h, w = flow.shape[:2]
        fx, fy = flow[:,:,0], flow[:,:,1]
        ang = np.arctan2(fy, fx) + np.pi
        v = np.sqrt(fx*fx+fy*fy)
        hsv = np.zeros((h, w, 3), np.uint8)
        hsv[...,0] = ang*(180/np.pi/2)
        hsv[...,1] = 255
    #    hsv[...,2] = np.minimum(v*16, 255)
        intensity = v > thres
    #    print intensity
        hsv[...,2] = np.minimum( intensity * 255, 255)
        bgr = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)
        return bgr
     
    def DrawLog( ax, log ):
        r,c = log.shape
        nb = c - 1
        for i in range(nb):
            ax.plot( log[:,0], log[:,1+i], label = "%5.2f to %5.2f" % ( ( 360.0/nb ) * i - 360.0/(2*nb),  ( 360.0/nb ) * (i + 1 ) - 360/(2*nb)) )
     
    def PrintException():
        exc_type, exc_obj, tb = sys.exc_info()
        f = tb.tb_frame
        lineno = tb.tb_lineno
        filename = f.f_code.co_filename
        linecache.checkcache(filename)
        line = linecache.getline(filename, lineno, f.f_globals)
        print 'EXCEPTION IN ({}, LINE {} "{}"): {}'.format(filename, lineno, line.strip(), exc_obj)
       
    if __name__ == '__main__':
        import sys
        print help_message
     
        for fn in sys.argv[1:]:
     
            print 'Opening file ' + fn
            cam = cv2.VideoCapture(fn)
     
            fig = plt.figure()
            ax = fig.add_subplot(1,1,1)
     
            ret, prev = cam.read()
            prevgray = cv2.cvtColor(prev, cv2.COLOR_BGR2GRAY)
            numBuckets = 8
            buckets = np.array( [ 0.0 ] * numBuckets  )
            log = np.zeros( ( 1000, numBuckets + 1 ) )
     
            print buckets
     
            frameNo = 0  
            fps = 10
            while True:
                ret, img = cam.read()
                frameNo = frameNo + 1
                try:
                    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
                    gray = cv2.blur( gray, (5,5))
                    flow = cv2.calcOpticalFlowFarneback(prevgray, gray, 0.5, 3, 15, 2, 5, 1,2)
                    UpdateBuckets( flow, buckets, 2.0 )
                    if ( ( frameNo % fps == 0 ) and ( frameNo / fps < len( log ) ) ):
                        log[frameNo / fps,:] = [ frameNo ] + [ b for b in buckets ]                    
     
                    if ( frameNo > 10000 ):
                        break
                    cv2.imshow('flow', draw_flow(gray, flow))
     
                    print 'Frame No ' + `frameNo` + " Buckets " + `buckets`
                    prevgray = gray            
                    ch = 0xFF & cv2.waitKey(5)
                    if ch == 27:
                        break
                except:
                    PrintException()
                    break
     
            log = log[0:frameNo/fps,:]
     
            DrawLog( ax, log )
            ax.legend(fancybox=True)
     
            plt.draw()
            plt.show()
            cv2.destroyAllWindows()

