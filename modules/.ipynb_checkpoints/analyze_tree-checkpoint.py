def curvline(start,end,rad,t=100,arrows=1,push=0.8):
    #Compute midpoint
    rad = rad/100.    
    x1, y1 = start
    x2, y2 = end
    y12 = (y1 + y2) / 2
    dy = (y2 - y1)
    cy = y12 + (rad) * dy
    #Prepare line
    tau = np.linspace(0,1,t)
    xsupport = np.linspace(x1,x2,t)
    ysupport = [(1-i)**2 * y1 + 2*(1-i)*i*cy + (i**2)*y2 for i in tau]
    #Create arrow data    
    arset = list(np.linspace(0,1,arrows+2))
    c = zip([xsupport[int(t*a*push)] for a in arset[1:-1]],
                      [ysupport[int(t*a*push)] for a in arset[1:-1]])
    dt = zip([xsupport[int(t*a*push)+1]-xsupport[int(t*a*push)] for a in arset[1:-1]],
                      [ysupport[int(t*a*push)+1]-ysupport[int(t*a*push)] for a in arset[1:-1]])
    arrowpath = zip(c,dt)
    return xsupport, ysupport, arrowpath

def plotcurv(start,end,rad,t=100,arrows=1,arwidth=.25,linewidth=1,color='black'):
    x, y, c = curvline(start,end,rad,t,arrows)
    plt.plot(x,y,'k-',color=cl,lw=linewidth)
    for d,dt in c:
        plt.arrow(d[0],d[1],dt[0],dt[1], shape='full', lw=0, 
                  length_includes_head=False, head_width=arwidth, color=cl)
    return c