"""
    Routine for plotting points and location solutions on a folium map or 
    in a matplot figure.
"""
import numpy as np
import folium
import webbrowser
import matplotlib.pyplot as plt

from tempfile import _get_candidate_names as tmpName

from pyloa.util import euclid, xy2lla
from pyloa.plane.parser import return_lola

__icon_colors = ['red', 'green', 'blue', 'black', 'purple', 'gray', 'darkred', 'darkblue',\
                 'darkgreen', 'darkpurple', 'cadetblue', 'lightgreen', 'lightgray',\
                 'lightred', 'lightblue', 'white', 'beige', 'pink', 'orange']

#----------------------------------------------------------------------

def plot_points( Y=None, X=None, lola=None, cust_id=None ):
    """
    Plots the customer points and the facilities. If geographical
    coordinates are available, a folium map is created. Otherwise,
    a matplotlib figure is displayed.
    
    Parameters
    ----------
    Y : None or numpy mx2 array of float
         Euclidian coordinates of the customer points
    X : None or numpy array of 2 float or px2 numpy array of float
         Euclidian coordinates of located facility/facilities
    lola : None or tuple of two lists of numpy arrays of float
         If not None, then lola[0] is the longitude data and lola[1] 
         the latitude data of the customer points
    cust_id : list or numpy array of str
        Names/id's of customer points
        
    Remark
    ------
    If longitude-latitude data are provided, then Y
    is assumed to be obtained from mapping these data
    to Euclidian coordinates using the most south-west
    point as origin.
    """
    lo, la, names = return_lola(with_names=True)
    if not lola is None: lo, la = lola 
    if not cust_id is None: names = cust_id
    if Y is None and (lo is None or la is None):
        # Nothing to plot
        exit()
    if lo is None or la is None:
        # No geographical data: Make matplotlib plot
        fig, ax = plt.subplots() 
        plt.axis([Y[:,0].min()-1,Y[:,0].max()+1,Y[:,1].min()-1,Y[:,1].max()+1])
        title = 'Customer points' if X is None else 'Customer and facility points'
        ax.set_title(title)
        ax.plot(Y[:,0],Y[:,1], 'bo')
        if not X is None:
            if len(X.shape)==1:
                ax.plot(X[0],X[1],'ro')
                for y in Y: plt.plot([y[0],X[0]],[y[1],X[1]],'k-')
            else:
                a = np.array([ np.argmin(np.fromiter((euclid(x, y) for x in X ),float)) for y in Y ])
                for j,x in enumerate(X):
                    customers = np.array(np.where( np.array(a)==j ))[0]
                    ax.plot(x[0],x[1],'ro')       
                    for y in Y[customers]: plt.plot([y[0],x[0]],[y[1],x[1]],'k-')
        plt.show()
    else:
        origin = (min(lo),min(la))
        m = folium.Map()
        m.fit_bounds([ (origin[1],origin[0]),(max(la),max(lo))])
        if X is None or type(X)==tuple or len(X.shape)==1 or X.shape[0]==1:
            for lat,lon,name in zip(la, lo, names):
                folium.Circle(radius=10,location=[lat, lon], popup=name,color="blue",fill=True).add_to(m)
            if not X is None:
                lon, lat = xy2lla(X[0],X[1],origin) if type(X)==tuple or len(X.shape)==1 \
                         else xy2lla(X[0][0],X[0][1], origin)
                folium.Marker( location=[lat, lon], popup="Facility location" ).add_to(m)
        else:
            # Assignment to the facilities
            n = Y.shape[0]
            p = X.shape[0]
            a = np.array([np.argmin(np.fromiter((euclid(X[j,:], Y[i,:]) for j in range(p)),float)) \
                for i in range(n) ])
            ncolors = len(__icon_colors)
            for j,x in enumerate(X):
                color = __icon_colors[j % ncolors]
                lon, lat = xy2lla(x[0],x[1], origin)
                folium.Marker( location=[lat, lon], popup="Facility location"+str(j+1),\
                               icon=folium.Icon(color=color) ).add_to(m)
                custs_j = np.where( a==j )[0]
                for i in custs_j:
                    folium.Circle(radius=10,location=[la[i],lo[i]],popup=names[i],\
                                  color=color,fill=True).add_to(m)
        print('Writing map to',__html_fil)
        m.save(__html_fil)
        webbrowser.open(__html_fil)
        
#----------------------------------------------------------------------
# HTML used for storing maps with folium    
__html_fil = next(tmpName())+'.html'
