import numpy as np
import os
import gemini3d.find as find
import gemini3d.read as read
import gemini3d.write as write
import h5py

RE = 6370e3

def makegrid(direc: str,
             dang=1.5,
             ltheta: int=16,
             lphi: int=16,
             return_grid=True,
             write_grid=False,
):
    #   dang (1,1) {mustBeNumeric} = 1.5 # ANGULAR RANGE TO COVER FOR THE CALCULATIONS (THIS IS FOR THE FIELD POINTS - SOURCE POINTS COVER ENTIRE GRID)

    assert (write_grid or return_grid),"Either 'return_grid' or 'write_grid' must be True"
    
    direc = os.path.expanduser(direc);
    assert os.path.isdir(direc), direc + " is not a directory"

    #SIMULATION METADATA
    cfg = read.config(direc)

    #WE ALSO NEED TO LOAD THE GRID FILE
    xg = read.grid(direc)
    print('Grid loaded')

    # lx1 = xg.lx(1);
    lx3 = xg['lx'][2]
    # lh=lx1;   %possibly obviated in this version - need to check
    if (lx3==1):
        flag2D=True
        print('2D meshgrid')
        #     x1=xg.x1(3:end-2);
        #     x2=xg.x2(3:end-2);
        #     x3=xg.x3(3:end-2);
        #     [X2,X1]=meshgrid(x2(:),x1(1:lh)');
    else:
        flag2D=False;
        print('3D meshgrid')
        #     x1=xg.x1(3:end-2);
        #     x2=xg.x2(3:end-2);
        #     x3=xg.x3(3:end-2);
        #     [X2,X1,X3]=meshgrid(x2(:),x1(1:lh)',x3(:));

    #TABULATE THE SOURCE OR GRID CENTER LOCATION
    if 'sourcemlon' not in cfg.keys():
        thdist = np.mean(xg['theta'])
        phidist = np.mean(xg['phi'])
    else:
        thdist= np.pi/2 - np.deg2rad(cfg['sourcemlat']);    #zenith angle of source location
        phidist= np.deg2rad(cfg['sourcemlon']);

    #FIELD POINTS OF INTEREST (CAN/SHOULD BE DEFINED INDEPENDENT OF SIMULATION GRID)
    # ltheta = 40
    if flag2D:
        lphi = 1
    else:
        # lphi = 40
        lphi = ltheta
    lr = 1

    thmin = thdist - np.deg2rad(dang);
    thmax = thdist + np.deg2rad(dang);
    phimin = phidist - np.deg2rad(dang);
    phimax = phidist + np.deg2rad(dang);

    theta = np.linspace(thmin,thmax,ltheta);
    if flag2D:
        phi = phidist;
    else:
        phi = np.linspace(phimin,phimax,lphi);

    r = RE*np.ones((ltheta,lphi));                          #use ground level for altitude for all field points

    phi,theta = np.meshgrid(phi,theta,indexing='ij');

    ## CREATE AN INPUT FILE OF FIELD POINTS
    gridsize = np.int32([lr,ltheta,lphi])
    mag = dict()
    mag['r'] = np.float32(r).ravel()
    mag['phi'] = np.float32(phi).ravel()
    mag['theta'] = np.float32(theta).ravel()
    mag['gridsize'] = gridsize

    mag['lpoints'] = np.prod(gridsize)

    if write_grid:
        filename = os.path.join(direc, "inputs/TESTmagfieldpoints.h5")
        print(f"Writing grid to {filename}")
        write.maggrid(filename, mag)
        print("Done!")

    if return_grid:
        return mag


def magframe(filename: str,**opts):
    """
    # example use
    # dat = gemini3d.read.magframe(filename)
    # dat = gemini3d.read.magframe(folder, "time", datetime)
    # dat = gemini3d.read.magframe(filename, "config", cfg)

    Translated from magframe.m 
    2022/07/05
    Spencer M Hatch
    
    Tweaks to deal with pygemini API idiodsyncracies.  Also force
      return with no value if binary files used (should be deprecated
      soon) -MZ.
    2022/7/7
    """

    # arguments
    # filename (1,1) string {mustBeNonzeroLengthText}
    # opts.time datetime {mustBeScalarOrEmpty} = datetime.empty
    # opts.cfg struct {mustBeScalarOrEmpty} = struct.empty
    # opts.gridsize (1,3) {mustBeInteger} = [-1,-1,-1]    # [lr,ltheta,lphi] grid sizes
    # end

    time = None
    gridsize = [-1,-1,-1]
    if 'time' in opts:
        time = opts['time']

    if 'gridsize' in opts:
        gridsize = opts['gridsize']

    # make sure to add the default directory where the magnetic fields are to
    # be found
    if os.path.isfile(filename):
        direc = os.path.dirname(os.path.dirname(filename));
    else:
        direc = os.path.dirname(filename);
    basemagdir = os.path.join(direc,"magfields")

    # find the actual filename if only the directory was given
    if not os.path.isfile(filename):
        if time is not None:
            filename = find.frame(basemagdir, opts['time'])

    # read the config file if one was not provided as input
    if 'config' in opts:
        cfg = opts['cfg']
    else:
        cfg = read.config(direc)

    # some times might not have magnetic field computed
    if len(str(filename)) == 0:
        print("SKIP: read.magframe %s", str(time))
        return

    # load and construct the magnetic field point grid
    assert cfg['file_format'] in ['.dat','.h5'], 'Unrecognized input field point file format {:s}'.format(cfg['file_format'])
    if cfg['file_format'] == '.dat':
        print("Have not implemented magframe in python for file_format=='dat'!")
        return;
        # fn = os.path.join(direc,'inputs/magfieldpoints.dat');
        # assert os.path.isfile(fn), fn + " not found"

        # fid = fopen(fn, 'r');
        # lpoints = fread(fid,1,'integer*4');
        # r = fread(fid,lpoints,'real*8');
        # theta = fread(fid,lpoints,'real*8');    #by default these are read in as a row vector, AGHHHH!!!!!!!!!
        # phi = fread(fid,lpoints,'real*8');
        # fclose(fid);
    elif cfg['file_format'] == '.h5':
        fn = os.path.join(direc,'inputs/magfieldpoints.h5');
        assert os.path.isfile(fn), fn + " not found"

        # matl = dict()
        # matl['lpoints'] = f3['lpoints'][()]
        # matl['gridsize'] = f3['gridsize'][:]
        # matl['r'] = f3['r'][:]
        # matl['theta'] = f3['theta'][:]
        # matl['phi'] = f3['phi'][:]

        with h5py.File(fn, "r") as h5f:
            lpoints   = h5f['lpoints'][()]
            gridsize  = h5f['gridsize'][:]
            r         = h5f['r'][:]
            theta     = h5f['theta'][:]
            phi       = h5f['phi'][:]

    # Reorganize the field points if the user has specified a grid size
    if any(elem < 0 for elem in gridsize):
        gridsize=[lpoints,1,1]    # just return a flat list if the user has not specified any gridding
        flatlist = True
    else:
        flatlist = False

    lr,ltheta,lphi = gridsize
    r = r.reshape(gridsize)
    theta = theta.reshape(gridsize)
    phi = phi.reshape(gridsize)

    # Sanity check the grid size and total number of grid points
    assert lpoints == np.prod(gridsize), 'Incompatible data size and grid specification...'

    # Create grid alt, magnetic latitude, and longitude (assume input points
    # have been permuted in this order)...
    mlat = 90-theta*180/np.pi
    mlon = phi*180/np.pi
    # breakpoint()
    dat = dict()
    if ~flatlist:   # we have a grid of points
        ilatsort = np.argsort(mlat[0,0,:])
        # [~,ilatsort]=sort(mlat(1,:,1))    #mlat runs against theta...
        dat['mlat'] = np.squeeze(mlat[0,0,ilatsort])
        # [~,ilonsort]=sort(mlon(1,1,:))
        ilonsort = np.argsort(mlon[0,:,0])
        dat['mlon'] = np.squeeze(mlon[0,ilonsort,0])
        dat['r'] = r[:,0,0]    # assume already sorted properly
    else:    # we have a flat list of points
        ilatsort = slice(0,lpoints)
        ilonsort = slice(0,lpoints)
        dat['mlat'] = mlat
        dat['mlon'] = mlon
        dat['r'] = r

    # allocate output arrays
    dat['Br'] = np.zeros((lr,ltheta,lphi))
    dat['Btheta'] = np.zeros((lr,ltheta,lphi))
    dat['Bphi'] = np.zeros((lr,ltheta,lphi))

    # Br
    if cfg['file_format'] == '.dat':
        return;
        #        fid = fopen(os.path.join(basemagdir,strcat(filename,".dat")),'r');
        # fid = fopen(filename,'r');
        # data = fread(fid,lpoints,'real*8');
    elif cfg['file_format'] == '.h5':
        data = np.array(h5py.File(filename, 'r')['/magfields/Br'])

    dat['Br'] = data.reshape([lr,ltheta,lphi]);
    if ~flatlist:
        dat['Br'] = dat['Br'][:,ilatsort,:]
        dat['Br'] = dat['Br'][:,:,ilonsort]

    # Btheta
    if cfg['file_format'] == '.dat':
        return;
        #data = fread(fid,lpoints,'real*8');
    elif cfg['file_format'] == '.h5':
        data = np.array(h5py.File(filename, 'r')['/magfields/Btheta'])

    dat['Btheta'] = data.reshape([lr,ltheta,lphi])
    if ~flatlist:
        dat['Btheta'] = dat['Btheta'][:,ilatsort,:]
        dat['Btheta'] = dat['Btheta'][:,:,ilonsort]

    # Bphi
    if cfg['file_format'] == '.dat':
        #data = fread(fid, lpoints,'real*8')
        return;
    if cfg['file_format'] == '.h5':
        data = np.array(h5py.File(filename, 'r')['/magfields/Bphi'])

    dat['Bphi'] =data.reshape([lr,ltheta,lphi])
    if ~flatlist:
        dat['Bphi'] = dat['Bphi'][:,ilatsort,:]
        dat['Bphi'] = dat['Bphi'][:,:,ilonsort]

    return dat
