import sys
sys.path.append('/scratch/gpfs/MIKHAILOVA/zl8336')
sys.path.append('/scratch/gpfs/MIKHAILOVA/zl8336/EM_analyzer')
import os
from typing import Optional,Union
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm,Normalize,XKCD_COLORS
from matplotlib.colorbar import Colorbar
from matplotlib.patches import Circle
from matplotlib.cm import ScalarMappable
from mpl_toolkits.axes_grid1 import make_axes_locatable
from mpl_toolkits.axes_grid1.axes_divider import AxesDivider
from EM_analyzer.plot.plot_basic import savefig,slice_between
from EM_analyzer.plot.plot_1D import plot_multiple_1D_fields
from EM_analyzer.plot.plot_basic import phase_amp_to_rgb
from EM_analyzer.pretreat_fields import get_closest_coordinate_id
color_list=list(XKCD_COLORS.keys())
working_dir=''
mpl.rcParams['xtick.labelsize'] = 20
mpl.rcParams['ytick.labelsize'] = 20


def add_colorbar(mappable,ax_cbar_divider:AxesDivider,label=''):
    cax: plt.Axes = ax_cbar_divider.append_axes("right", size="90%",pad=1.6)
    cbar=plt.colorbar(mappable, cax=cax)
    cax.set_xlabel(label,fontsize=25, labelpad=10)
    cax.yaxis.set_tick_params(
        direction="out",length=6,  width=1, colors='black', grid_color='black', grid_alpha=0.5,
        left=False,right=False,labelleft=False,labelright=True)
    return cbar


def generate_side_panel_axes(
    ax_main_height=8.0,ax_main_width=8.0,ax_main_projection='rectilinear',
    side_panel_size=3.0,ax_cbar_width = 4,
    generate_ax_panel_top=False,generate_ax_panel_right=False,generate_ax_panel_bottom=False,generate_ax_panel_left=False,
    generate_ax_legend=False,
    ):
    """_summary_
    Args:
        ax_main_height (float, optional): _description_. Defaults to 8. unit: inch.
        ax_main_width (float, optional): _description_. Defaults to 8. unit: inch.
        side_panel_size (float, optional): _description_. Defaults to 3. unit: inch.
        ax_cbar_width (float, optional): _description_. Defaults to 0.3. unit: inch.
        ax_main_projection (str, optional): _description_. Defaults to 'rectilinear'.
        generate_ax_panel_top (bool, optional): _description_. Defaults to False.
        generate_ax_panel_right (bool, optional): _description_. Defaults to False.
        generate_ax_panel_bottom (bool, optional): _description_. Defaults to False.
        generate_ax_panel_left (bool, optional): _description_. Defaults to False.
    """
    figure_base_left = 1.5   #unit: inch
    figure_base_right = 1.5   #unit: inch
    figure_base_bottom = 1.5   #unit: inch
    figure_base_top = 1.5   #unit: inch
    ax_cbar_gap = 0.5 #unit: inch
    ax_cbar_height = ax_main_height #unit: inch
    figure_width = figure_base_left + ax_main_width + generate_ax_panel_left*side_panel_size+generate_ax_panel_right*side_panel_size + ax_cbar_gap +ax_cbar_width +figure_base_right #unit: inch
    figure_height = figure_base_bottom + ax_main_height + generate_ax_panel_top*side_panel_size+generate_ax_panel_bottom*side_panel_size +figure_base_top #unit: inch
    fig = plt.figure(figsize=(figure_width,figure_height),dpi=100)
    print('figure size=',fig.get_size_inches(),'inch')
    ax_main_left=figure_base_left+generate_ax_panel_left*side_panel_size   #unit: inch
    ax_main_bottom=figure_base_bottom+generate_ax_panel_bottom*side_panel_size   #unit: inch
    ax_main: plt.Axes = fig.add_axes([ax_main_left/figure_width, ax_main_bottom/figure_height, ax_main_width/figure_width, ax_main_height/figure_height],projection=ax_main_projection)
    print('ax_main position=',ax_main.get_position())
    ax_dict={'fig':fig,'ax_main':ax_main}
    if generate_ax_panel_top:
        ax_panel_top_left = ax_main_left
        ax_panel_top_bottom = ax_main_bottom + ax_main_height
        ax_panel_top :plt.Axes = fig.add_axes([ax_panel_top_left/figure_width, ax_panel_top_bottom/figure_height, ax_main_width/figure_width, side_panel_size/figure_height],sharex=ax_main)
        ax_dict['ax_panel_top']=ax_panel_top
    if generate_ax_panel_right:
        ax_panel_right_left = ax_main_left + ax_main_width
        ax_panel_right_bottom = ax_main_bottom
        ax_panel_right :plt.Axes = fig.add_axes([ax_panel_right_left/figure_width, ax_panel_right_bottom/figure_height, side_panel_size/figure_width, ax_main_height/figure_height],sharey=ax_main)
        ax_dict['ax_panel_right']=ax_panel_right
    if generate_ax_panel_bottom:
        ax_panel_bottom_left = ax_main_left
        ax_panel_bottom_bottom = figure_base_bottom
        ax_panel_bottom :plt.Axes = fig.add_axes([ax_panel_bottom_left/figure_width, ax_panel_bottom_bottom/figure_height, ax_main_width/figure_width, side_panel_size/figure_height],sharex=ax_main)
        ax_dict['ax_panel_bottom']=ax_panel_bottom
    if generate_ax_panel_left:
        ax_panel_left_left = figure_base_left
        ax_panel_left_bottom = ax_main_bottom
        ax_panel_left :plt.Axes = fig.add_axes([ax_panel_left_left/figure_width, ax_panel_left_bottom/figure_height, side_panel_size/figure_width, ax_main_height/figure_height],sharey=ax_main)
        ax_dict['ax_panel_left']=ax_panel_left
    
    
    ax_cbar_left = ax_main_left + ax_main_width + generate_ax_panel_right*side_panel_size + ax_cbar_gap 
    ax_cbar_bottom = ax_main_bottom
    ax_cbar :plt.Axes = fig.add_axes([ax_cbar_left/figure_width, ax_cbar_bottom/figure_height, ax_cbar_width/figure_width, ax_cbar_height/figure_height])
    ax_cbar.axis("off")
    ax_cbar_divider = make_axes_locatable(ax_cbar)
    ax_dict['ax_cbar']=ax_cbar
    ax_dict['ax_cbar_divider']=ax_cbar_divider
    if generate_ax_legend:
        ax_legend_left=ax_main_left + ax_main_width
        ax_legend_bottom=ax_main_bottom + ax_main_height
        ax_legend :plt.Axes = fig.add_axes([ax_legend_left/figure_width,ax_legend_bottom/figure_height,side_panel_size/figure_width,side_panel_size/figure_height])
        ax_legend.set_facecolor((1, 1, 1, 0))
        ax_legend.set_xticks([])
        ax_legend.set_yticks([])
        ax_dict['ax_legend']=ax_legend
    return ax_dict
    
def plot_2D_field_with_side_panel(
    field:np.ndarray,
    field_top_panel_dict:Optional[dict]=None,
    field_bottom_panel_dict:Optional[dict]=None,
    field_right_panel_dict:Optional[dict]=None,
    field_left_panel_dict:Optional[dict]=None,
    ax_dict:Optional[dict]=None,
    x_coordinate=[0],y_coordinate=[0],
    vmin:Optional[float]=None,vmax:Optional[float]=None,scale='linear',cmap='seismic',
    alpha=1.0,aspect=1.0,
    xmin:Optional[float]=None,xmax:Optional[float]=None,ymin:Optional[float]=None,ymax:Optional[float]=None,
    label=r'$a=\frac{E}{E_c}=\frac{B}{B_c}$',xlabel=r'$\frac{x}{\lambda_0}$',ylabel=r'$\frac{y}{\lambda_0}$',
    step_x=1,step_y=1,
    plot_colorbar=True,
    return_fig=True,name='',working_dir='.',
    ):
    r"""_summary_
    All variables should be normalized in advance
    Args:
        field (np.ndarray): _description_
        field_top_panel_dict,field_bottom_panel_dict,field_right_panel_dict,field_left_panel_dict: 
        {
            'field_dict_list':[{'field':field,'linestyle':'-','label':None,'color':None},...] ,
            'axhline_dict_list':[{'y':y,'linestyle':'--','label':None,'color':None},...],
            'axvline_dict_list':[{'x':x,'linestyle':'--','label':None,'color':None},...],
            'vmin':float=vmin,
            'vmax':float=vmax,
            'scale':str | ScaleBase='linear',
            'label':str=label,
            'title':str=None,
            'plot_legend':bool=False,
        }
        ax_dict:
        {
            'fig':fig,
            'ax_main':ax_main,   #must be provided
            'ax_panel_top':ax_panel_top,   #can be None
            'ax_panel_right':ax_panel_right,   #can be None
            'ax_panel_bottom':ax_panel_bottom,   #can be None
            'ax_panel_left':ax_panel_left,   #can be None
            'ax_cbar_0':ax_cbar_0,   #can be None
        }
        x_coordinate (_type_, optional): _description_. Defaults to x_coordinate.
        y_coordinate (_type_, optional): _description_. Defaults to y_coordinate.
        vmin (_type_, optional): _description_. Defaults to -1.
        vmax (_type_, optional): _description_. Defaults to 1.
        cmap (str, optional): _description_. Defaults to 'seismic'.
        aspect: (str or float, optional): _description_. Defaults to 1 (equal aspect ratio).
        label (regexp, optional): _description_. Defaults to ''.
        return_fig (bool, optional): _description_. Defaults to True.
        name (str, optional): _description_. Defaults to ''.
        working_dir (str, optional): _description_. Defaults to '.'.
    Returns:
        _type_: _description_
    """
    # Subset data for plotting
    x_coordinate=np.asarray(x_coordinate).flatten()
    y_coordinate=np.asarray(y_coordinate).flatten()
    x_slice=slice_between(x_coordinate, xmin, xmax, step=step_x)
    y_slice=slice_between(y_coordinate, ymin, ymax, step=step_y)
    x_coordinate_show=x_coordinate[x_slice]
    y_coordinate_show=y_coordinate[y_slice]
    xmin=x_coordinate_show[0]
    xmax=x_coordinate_show[-1]
    ymin=y_coordinate_show[0]
    ymax=y_coordinate_show[-1]
    l_x=xmax-xmin
    l_y=ymax-ymin
    field_show=np.asarray(field)[x_slice, y_slice]
    
    if vmin is None:
        vmin=np.nanmin(field_show)
    if vmax is None:
        vmax=np.nanmax(field_show)
    assert scale in ['linear','log']
    if scale=='log':
        norm=LogNorm(vmin=max(vmin,1e-10),vmax=vmax,clip=True)
    else:
        norm=Normalize(vmin=vmin,vmax=vmax,clip=True)

    if ax_dict is None:
        ax_main_height=8   #unit: inch
        if type(aspect)==str:
            aspect=l_x/l_y
        ax_main_width=min(ax_main_height*l_x/(l_y*aspect), ax_main_height*3)   #unit: inch
        ax_dict=generate_side_panel_axes(
            ax_main_height=ax_main_height,ax_main_width=ax_main_width,
            generate_ax_panel_top=field_top_panel_dict is not None,
            generate_ax_panel_right=field_right_panel_dict is not None,
            generate_ax_panel_bottom=field_bottom_panel_dict is not None,
            generate_ax_panel_left=field_left_panel_dict is not None,
            generate_ax_legend=False,
        )
    fig:plt.Figure=ax_dict['fig']
    ax_main:plt.Axes=ax_dict['ax_main']
    ax_cbar_divider:AxesDivider=ax_dict['ax_cbar_divider']
    pcm = ax_main.pcolormesh(
        x_coordinate_show, 
        y_coordinate_show, 
        field_show.T, 
        cmap=cmap, shading='auto',norm=norm, alpha=alpha,
        )
    ax_main.set_xlabel(xlabel,fontsize=25)
    ax_main.set_ylabel(ylabel,fontsize=25, rotation=0)
    ax_main.xaxis.set_label_position('bottom' if field_bottom_panel_dict is None else 'top')
    ax_main.yaxis.set_label_position('left' if field_left_panel_dict is None else 'right')
    ax_main.xaxis.set_tick_params(labelrotation=0,top=False,bottom=False,labeltop=field_top_panel_dict is None,labelbottom=field_bottom_panel_dict is None)
    ax_main.yaxis.set_tick_params(labelrotation=0,left=False,right=False,labelleft=field_left_panel_dict is None,labelright=field_right_panel_dict is None)
    ax_main.set_xlim(xmin,xmax)
    ax_main.set_ylim(ymin,ymax)
    ax_main.set_aspect('auto')
    ax_main.grid(True, alpha=0.4)
    ax_dict['ax_main']=ax_main
    if field_top_panel_dict is not None:
        ax_panel_top:plt.Axes= ax_dict['ax_panel_top']
        ax_panel_top= plot_multiple_1D_fields(
            coordinate_direction='vertical',
            coordinate=x_coordinate,
            ax=ax_panel_top,
            field_dict_list=field_top_panel_dict.get('field_dict_list',None),
            axhline_dict_list=field_top_panel_dict.get('axhline_dict_list',None),
            axvline_dict_list=field_top_panel_dict.get('axvline_dict_list',None),
            xmin=xmin,xmax=xmax,
            ymin=field_top_panel_dict.get('vmin',vmin),ymax=field_top_panel_dict.get('vmax',vmax),
            xlabel=xlabel,ylabel=field_top_panel_dict.get('label',label),
            xscale='linear',yscale=field_top_panel_dict.get('scale','linear'),
            plot_legend=field_top_panel_dict.get('plot_legend',False),
            name=field_top_panel_dict.get('title',None),
            return_fig=True,
        )['ax_main']
        ax_panel_top.xaxis.set_label_position('top')
        ax_panel_top.xaxis.set_tick_params(top=True,bottom=False,labeltop=True,labelbottom=False)
        ax_panel_top.yaxis.set_tick_params(left=True,right=True,labelleft=True,labelright=True)
        ax_dict['ax_panel_top']=ax_panel_top
    if field_bottom_panel_dict is not None:
        ax_panel_bottom:plt.Axes=ax_dict['ax_panel_bottom']
        ax_panel_bottom= plot_multiple_1D_fields(
            coordinate_direction='vertical',
            coordinate=x_coordinate,
            ax=ax_panel_bottom,
            field_dict_list=field_bottom_panel_dict.get('field_dict_list',None),
            axhline_dict_list=field_bottom_panel_dict.get('axhline_dict_list',None),
            axvline_dict_list=field_bottom_panel_dict.get('axvline_dict_list',None),
            xmin=xmin,xmax=xmax,
            ymin=field_bottom_panel_dict.get('vmin',vmin),ymax=field_bottom_panel_dict.get('vmax',vmax),
            xlabel=xlabel,ylabel=field_bottom_panel_dict.get('label',label),
            xscale='linear',yscale=field_bottom_panel_dict.get('scale','linear'),
            plot_legend=field_bottom_panel_dict.get('plot_legend',False),
        )['ax_main']
        ax_panel_bottom.xaxis.set_label_position('bottom')
        ax_panel_bottom.xaxis.set_tick_params(bottom=True,top=True,labelbottom=True,labeltop=True)
        ax_panel_bottom.yaxis.set_tick_params(left=True,right=True,labelleft=True,labelright=True)
        ax_dict['ax_panel_bottom']=ax_panel_bottom
    
    if field_right_panel_dict is not None:
        ax_panel_right:plt.Axes=ax_dict['ax_panel_right']
        ax_panel_right= plot_multiple_1D_fields(
            coordinate_direction='horizontal',
            coordinate=y_coordinate,
            ax=ax_panel_right,
            field_dict_list=field_right_panel_dict.get('field_dict_list',None),
            axhline_dict_list=field_right_panel_dict.get('axhline_dict_list',None),
            axvline_dict_list=field_right_panel_dict.get('axvline_dict_list',None),
            xmin=field_right_panel_dict.get('vmin',vmin),xmax=field_right_panel_dict.get('vmax',vmax),
            ymin=ymin,ymax=ymax,
            xlabel=field_right_panel_dict.get('label',label),ylabel=ylabel,
            xscale=field_right_panel_dict.get('scale','linear'),yscale='linear',
            plot_legend=field_right_panel_dict.get('plot_legend',False),
            name=field_right_panel_dict.get('title',None),
            return_fig=True,
        )['ax_main']
        ax_panel_right.yaxis.set_label_position('right')
        ax_panel_right.xaxis.set_tick_params(bottom=True,top=True,labelbottom=True,labeltop=True)
        ax_panel_right.yaxis.set_tick_params(left=False,right=True,labelleft=False,labelright=True,rotation=0)
        ax_dict['ax_panel_right']=ax_panel_right
    if field_left_panel_dict is not None:
        ax_panel_left:plt.Axes=ax_dict['ax_panel_left']
        ax_panel_left= plot_multiple_1D_fields(
            coordinate_direction='horizontal',
            coordinate=y_coordinate,
            ax=ax_panel_left,
            field_dict_list=field_left_panel_dict.get('field_dict_list',None),
            axhline_dict_list=field_left_panel_dict.get('axhline_dict_list',None),
            axvline_dict_list=field_left_panel_dict.get('axvline_dict_list',None),
            xmin=field_left_panel_dict.get('vmin',vmin),xmax=field_left_panel_dict.get('vmax',vmax),
            ymin=ymin,ymax=ymax,
            xlabel=field_left_panel_dict.get('label',label),ylabel=ylabel,
            xscale=field_left_panel_dict.get('scale','linear'),yscale='linear',
            plot_legend=field_left_panel_dict.get('plot_legend',False),
            name=field_left_panel_dict.get('title',None),
            return_fig=True,
        )['ax_main']
        #ax_panel_left.invert_xaxis()
        ax_panel_left.xaxis.set_label_position('top')
        ax_panel_left.xaxis.set_tick_params(bottom=True,top=True,labelbottom=True,labeltop=True,rotation=0)
        ax_panel_left.yaxis.set_tick_params(left=True,right=False,labelleft=True,labelright=False,rotation=0)
        ax_dict['ax_panel_left']=ax_panel_left

    if plot_colorbar:
        add_colorbar(pcm,ax_cbar_divider,label=label)
    fig.suptitle(name,fontsize=25)
    if return_fig:
        return ax_dict
    else:
        return savefig(fig=ax_dict['fig'],fig_path=os.path.join(working_dir,'%s.png' %(name)))

def plot_2D_field(
    field:np.ndarray,
    ax_dict:Optional[dict]=None,
    x_coordinate=[0],y_coordinate=[0],
    threshold:Optional[float]=None,vmin:Optional[float]=None,vmax:Optional[float]=None,
    scale= 'linear',
    xmin:Optional[float]=None,xmax:Optional[float]=None,ymin:Optional[float]=None,ymax:Optional[float]=None,
    cmap='seismic',alpha:float=1.0,aspect=1.0,
    label=r'$a=\frac{E}{E_c}=\frac{B}{B_c}$',xlabel=r'$\frac{x}{\lambda_0}$',ylabel=r'$\frac{y}{\lambda_0}$',
    step_x=1,step_y=1,
    plot_profile_x=True,profile_at_x:Optional[float|list[float]]=None,ax_profile_x_at='left',
    plot_profile_y=True,profile_at_y:Optional[float|list[float]]=None,ax_profile_y_at='top',
    plot_colorbar=True,
    return_fig=True,name='',working_dir='.'):
    """
    All variables should be normalized in advance
        ax_dict:
        {
            'fig':fig,
            'ax_main':ax_main,   #must be provided
            'ax_panel_top':ax_panel_top,   #can be None
            'ax_panel_right':ax_panel_right,   #can be None
            'ax_panel_bottom':ax_panel_bottom,   #can be None
            'ax_panel_left':ax_panel_left,   #can be None
            'ax_cbar_divider':ax_cbar_divider,
        }
        profile_at_x: float or list[float], optional
            Single list of x positions for vertical profiles (along y-direction)
        profile_at_y: float or list[float], optional
            Single list of y positions for horizontal profiles (along x-direction)
    """
    field=np.asarray(field)
    x_coordinate=np.asarray(x_coordinate).flatten()
    y_coordinate=np.asarray(y_coordinate).flatten()
    n_x=x_coordinate.size
    n_y=y_coordinate.size
    assert field.shape==(n_x,n_y),f'field shape {field.shape} does not match coordinate sizes {(n_x,n_y)}'
    if threshold!=None:
        field_masked = np.where(np.abs(field) >= threshold, field, np.nan)
    else:
        field_masked=field
    assert scale in ['linear','log']

    field_max_id=tuple(np.asarray(np.where(np.abs(field)==np.nanmax(np.abs(field))),dtype=np.int32)[:,0])   #field_max_id=(x_id,y_id)

    # Build field_dict_list for multiple vertical profiles (top panel)
    # Create colormap and normalizers for colors
    # Normalize based on index values for color mapping

    # `plot_legend` is disabled because the side-panel legend text is too small
    # to read; instead each panel's title carries the slice coordinates.
    profile_x_dict={'plot_legend':False,'scale':scale}
    profile_y_dict={'plot_legend':False,'scale':scale}
    if plot_profile_x:
        field_dict_list_x = []
        axvline_dict_list = []
        if profile_at_x is None:
            profile_at_x_id_list=[field_max_id[0]]
        else:
            profile_at_x=np.asarray(profile_at_x).flatten()
            profile_at_x_id_list=get_closest_coordinate_id(coordinate=x_coordinate,pos=profile_at_x)
        # x_profile_norm = Normalize(vmin=np.min(profile_at_x_id_list), vmax=np.max(profile_at_x_id_list))
        # x_profile_color=plt.get_cmap('tab10_r')
        for i, x_id in enumerate(profile_at_x_id_list):
            #color = x_profile_color(x_profile_norm(x_id))
            color = color_list[i % len(color_list)]
            field_dict_list_x.append({
                'field': field[x_id, :],
                'label': f'{xlabel}={x_coordinate[x_id]:.2f}',
                'color': color,
            })
            axvline_dict_list.append({
                'x': x_coordinate[x_id],
                'color': color,
            })
        assert ax_profile_x_at in ['left','right']
        profile_x_dict['field_dict_list']=field_dict_list_x
        profile_y_dict['axvline_dict_list']=axvline_dict_list
        profile_x_dict['title']=(
            f'{xlabel} = '
            + ', '.join(f'{x_coordinate[x_id]:.2f}' for x_id in profile_at_x_id_list)
        )
        # Turn the panel legend back on only when there are multiple slices —
        # a single slice is already fully described by the title.
        if len(profile_at_x_id_list) > 1:
            profile_x_dict['plot_legend']=True

    if plot_profile_y:
        field_dict_list_y = []
        axhline_dict_list = []
        # Convert profile_at_y to array
        if profile_at_y is None:
            profile_at_y_id_list=[field_max_id[1]]
        else:
            profile_at_y=np.asarray(profile_at_y).flatten()
            profile_at_y_id_list=get_closest_coordinate_id(coordinate=y_coordinate,pos=profile_at_y)
        # y_profile_norm = Normalize(vmin=np.min(profile_at_y_id_list), vmax=np.max(profile_at_y_id_list))
        # y_profile_color=plt.get_cmap('tab10')
        for i,y_id in enumerate(profile_at_y_id_list):
            #color = y_profile_color(y_profile_norm(y_id))
            color = color_list[(i+20) % len(color_list)]
            field_dict_list_y.append({
                'field': field[:, y_id],
                'label': f'{ylabel}={y_coordinate[y_id]:.2f}',
                'color': color,
            })
            axhline_dict_list.append({
                'y': y_coordinate[y_id],
                'color': color,
            })
        assert ax_profile_y_at in ['top','bottom']
        profile_x_dict['axhline_dict_list']=axhline_dict_list
        profile_y_dict['field_dict_list']=field_dict_list_y
        profile_y_dict['title']=(
            f'{ylabel} = '
            + ', '.join(f'{y_coordinate[y_id]:.2f}' for y_id in profile_at_y_id_list)
        )
        if len(profile_at_y_id_list) > 1:
            profile_y_dict['plot_legend']=True



    ax_dict=plot_2D_field_with_side_panel(
        field=field_masked,
        field_top_panel_dict=profile_y_dict if plot_profile_y and ax_profile_y_at=='top' else None,
        field_bottom_panel_dict=profile_y_dict if plot_profile_y and ax_profile_y_at=='bottom' else None,
        field_left_panel_dict=profile_x_dict if plot_profile_x and ax_profile_x_at=='left' else None,
        field_right_panel_dict=profile_x_dict if plot_profile_x and ax_profile_x_at=='right' else None,
        ax_dict=ax_dict,
        x_coordinate=x_coordinate,y_coordinate=y_coordinate,
        vmin=vmin,vmax=vmax,cmap=cmap,scale=scale,alpha=alpha,
        xmin=xmin,xmax=xmax,ymin=ymin,ymax=ymax,
        label=label,xlabel=xlabel,ylabel=ylabel,
        step_x=step_x,step_y=step_y,
        plot_colorbar=plot_colorbar,aspect=aspect,
        return_fig=True,name=name,working_dir=working_dir,
    )
    if plot_profile_x or plot_profile_y:
        ax_dict['ax_main']=plot_multiple_1D_fields(
            coordinate=x_coordinate,
            axhline_dict_list=axhline_dict_list,
            axvline_dict_list=axvline_dict_list,
            plot_legend=False,set_label=False,return_fig=True,
            ax=ax_dict['ax_main'],xmin=xmin,xmax=xmax,ymin=ymin,ymax=ymax,xlabel=xlabel,ylabel=ylabel,
        )['ax_main']
    if return_fig:
        return ax_dict
    else:
        return savefig(fig=ax_dict['fig'],fig_path=os.path.join(working_dir,'%s.png' %(name)))


def plot_2D_spectrum(
    spectrum_square:np.ndarray,
    ax_dict:Optional[dict]=None,
    kx_coordinate=[0],ky_coordinate=[0],
    threshold:Optional[float]=None,vmin:Optional[float]=1e-6,vmax:Optional[float]=1.1,scale='log',
    kxmin:Optional[float]=0,kxmax:Optional[float]=None,kymin:Optional[float]=None,kymax:Optional[float]=None,
    cmap='hot',alpha:float=1.0,aspect=1.0,
    label='I(kx,ky)',xlabel='kx',ylabel='ky',
    step_x=1,step_y=1,
    plot_profile_x=True,profile_at_x:Optional[float|list[float]]=1,profile_at_y:Optional[float|list[float]]=0,
    plot_colorbar=True,
    return_fig=True,name='',working_dir='.'):
    return plot_2D_field(
        field=spectrum_square,
        ax_dict=ax_dict,
        x_coordinate=kx_coordinate,y_coordinate=ky_coordinate,
        threshold=threshold,vmin=vmin,vmax=vmax,scale=scale,
        xmin=kxmin,xmax=kxmax,ymin=kymin,ymax=kymax,
        cmap=cmap,alpha=alpha,aspect=aspect,
        label=label,xlabel=xlabel,ylabel=ylabel,
        step_x=step_x,step_y=step_y,
        plot_profile_x=plot_profile_x,profile_at_x=profile_at_x,profile_at_y=profile_at_y,
        plot_colorbar=plot_colorbar,
        return_fig=return_fig,name=name,working_dir=working_dir,
    )


def plot_multiple_2D_fields(
    field_dict_list:list[dict],x_coordinate,y_coordinate,
    xmin:Optional[float]=None,xmax:Optional[float]=None,ymin:Optional[float]=None,ymax:Optional[float]=None,
    label=r'$a=\frac{E}{E_c}=\frac{B}{B_c}$',xlabel=r'$\frac{x}{\lambda_0}$',ylabel=r'$\frac{y}{\lambda_0}$',
    step_x:int=1,step_y:int=1,
    return_fig=True,name='',working_dir='.',
    aspect=1.0,
    ):
    """
    Args:
        field_dict_list=
            [
                {
                    'field':field,
                    'vmin':None,'vmax':None,'cmap':'seismic',
                    'label':label,
                    'threshold':None,
                    'plot_profile_x':False,'profile_at_x':Optional[int]=None,'profile_at_y':Optional[int]=None,
                    'plot_colorbar':True,
                    'alpha':1.0,
                    'scale':'linear',
                },
            ...
            ] 
        x_coordinate (_type_): _description_
        y_coordinate (_type_): _description_
        xmin (Optional[float], optional): _description_. Defaults to None.
        xmax (Optional[float], optional): _description_. Defaults to None.
        ymin (Optional[float], optional): _description_. Defaults to None.
        ymax (Optional[float], optional): _description_. Defaults to None.
        label (regexp, optional): _description_. Defaults to ''.
        xlabel (regexp, optional): _description_. Defaults to ''.
        ylabel (regexp, optional): _description_. Defaults to ''.
        step_x (int, optional): Step size for x-axis. Defaults to 1.
        step_y (int, optional): Step size for y-axis. Defaults to 1.
        name (str, optional): _description_. Defaults to ''.
    """
    plot_profile_x=any([field_dict.get('plot_profile_x',False) for field_dict in field_dict_list])
    field_num=len(field_dict_list)
    l_x=xmax-xmin
    l_y=ymax-ymin
    ax_main_height=6   #unit: inch
    ax_main_width=ax_main_height*l_x/l_y   #unit: inch
    ax_dict=generate_side_panel_axes(
        ax_main_height=ax_main_height,ax_main_width=ax_main_width,
        generate_ax_panel_top=plot_profile_x,
        generate_ax_panel_left=plot_profile_x,
    )
    for i, field_dict in enumerate(field_dict_list):
        ax_dict=plot_2D_field(
            field=field_dict['field'],
            ax_dict=ax_dict,
            x_coordinate=x_coordinate,y_coordinate=y_coordinate,
            threshold=field_dict.get('threshold',None),vmin=field_dict.get('vmin',None),vmax=field_dict.get('vmax',None),
            scale=field_dict.get('scale','linear'),
            xmin=xmin,xmax=xmax,ymin=ymin,ymax=ymax,
            cmap=field_dict.get('cmap','seismic'),alpha=field_dict.get('alpha',1.0),aspect=aspect,
            label=field_dict.get('label',label),xlabel=xlabel,ylabel=ylabel,
            return_fig=True,
            step_x=step_x,step_y=step_y,
            plot_profile_x=field_dict.get('plot_profile_x',False),
            profile_at_x=field_dict.get('profile_at_x',None),
            profile_at_y=field_dict.get('profile_at_y',None),
            plot_colorbar=field_dict.get('plot_colorbar',True),
        )
    fig:plt.Figure=ax_dict['fig']
    fig.suptitle(name,fontsize=25)
    if return_fig:
        return ax_dict
    else:
        return savefig(fig=fig,fig_path=os.path.join(working_dir,'%s.png' %(name)))


def plot_polar_field(
    field:np.ndarray,
    r_coordinate:np.ndarray,a_coordinate:np.ndarray,
    ax_dict:Optional[dict]=None,
    r_min:Optional[float]=None,r_max:Optional[float]=None,
    a_min:Optional[float]=None,a_max:Optional[float]=None,
    v_min:Optional[float]=-1,v_max:Optional[float]=1,
    cmap='seismic',alpha=1.0,scale='log',
    label=r'$a=\frac{E}{E_c}=\frac{B}{B_c}$',r_label='',a_label='θ',
    plot_colorbar=True,
    return_fig=True,name='',working_dir='.',
):
    """
    All variables should be normalized in advance
    
    Parameters
    ----------
    field : np.ndarray
        2D array of shape (N_a, N_r) representing the field values in polar coordinates.
    r_coordinate : np.ndarray
        1D array of shape (N_r,) representing the radial coordinates.
    a_coordinate : np.ndarray
        1D array of shape (N_a,) representing the angular coordinates (in radians).
    ax_dict : dict, optional
        Dictionary containing axes for plotting. If None, new axes will be created using `generate_side_panel_axes`.
    r_min, r_max : float, optional
        Minimum and maximum values for the radial coordinate. If None, they will be determined from `r_coordinate`.
    a_min, a_max : float, optional
        Minimum and maximum values for the angular coordinate (in radians). If None, they will be set to -π and π, respectively.
    v_min, v_max : float, optional
        Minimum and maximum values for the field values. If None, they will be determined from `field`.
    cmap : str, optional
        Colormap name for plotting the field. Defaults to 'seismic'.
    alpha : float, optional
        Alpha value for the colormap. Defaults to 1.0 (fully opaque).
    scale : str, optional
        Scale for color normalization. Can be 'linear' or 'log'. Defaults to 'log'.
    label : str, optional
        Label for the colorbar. Defaults to a LaTeX string representing the normalized field.
    r_label : str, optional
        Label for the radial coordinate. Defaults to an empty string.
    a_label : str, optional
        Label for the angular coordinate. Defaults to 'θ'.
    plot_colorbar : bool, optional
        Whether to plot the colorbar. Defaults to True.
    return_fig : bool, optional
        Whether to return the figure and axes dictionary. If False, the figure will be saved to a file. Defaults to True.
    name : str, optional
        Name for the plot title and saved file. Defaults to an empty string.
    working_dir : str, optional
        Directory to save the figure if `return_fig` is False. Defaults to the current directory
    
    """
    field=np.asarray(field)
    a_coordinate=np.array(a_coordinate).flatten()
    r_coordinate=np.array(r_coordinate).flatten()
    N_a=a_coordinate.size
    N_r=r_coordinate.size
    assert field.shape==(N_a,N_r)
    if r_min is None:
        r_min=0
    if r_max is None:
        r_max=np.nanmax(r_coordinate)
    if a_min is None:
        a_min=-np.pi
    if a_max is None:
        a_max=np.pi
    if v_min is None:
        v_min=np.nanmin(field)
    if v_max is None:
        v_max=np.nanmax(field)
    assert scale in ['linear','log']
    if scale=='log':
        norm=LogNorm(vmin=max(v_min,1e-10),vmax=v_max,clip=True)
    else:
        norm=Normalize(vmin=v_min,vmax=v_max,clip=True)
    if ax_dict is None:
        ax_dict=generate_side_panel_axes(
            ax_main_height=6,ax_main_width=6,
            ax_main_projection='polar'
        )
    fig:plt.Figure=ax_dict['fig']
    ax_main:plt.Axes=ax_dict['ax_main']
    ax_cbar_divider:AxesDivider=ax_dict['ax_cbar_divider']
    pcm=ax_main.pcolormesh(a_coordinate, r_coordinate, field.T, cmap=cmap, shading='auto', norm=norm, alpha=alpha)
    ax_main.set_xlabel(a_label, fontsize=25)
    ax_main.set_ylabel(r_label, fontsize=25, labelpad=20)
    ax_main.set_thetamin(np.degrees(a_min))
    ax_main.set_thetamax(np.degrees(a_max))
    ax_main.set_rlim(r_min, r_max)
    ax_main.set_theta_direction(1)
    ax_main.set_theta_zero_location('E')
    if plot_colorbar:
        add_colorbar(pcm, ax_cbar_divider, label=label)
    ax_main.grid(True, linestyle='--', alpha=0.7)
    fig.suptitle(name, fontsize=25)
    if return_fig:
        return ax_dict
    else:
        return savefig(fig=fig, fig_path=os.path.join(working_dir, f'{name}.png'))


def plot_quiver_field(
    x_coordinate: np.ndarray,
    y_coordinate: np.ndarray,
    Vx: np.ndarray,
    Vy: np.ndarray,
    A: Optional[np.ndarray] = None,
    B: Optional[np.ndarray] = None,
    ax_dict:Optional[dict]=None,
    step_x: int = 1,step_y: int = 1,
    aspect: Optional[Union[str, float]] = 'auto',
    threshold: float = 0.0,
    Bmin: Optional[float] = None,
    Bmax: Optional[float] = None,
    xmin: Optional[float] = None,xmax: Optional[float] = None,ymin: Optional[float] = None,ymax: Optional[float] = None,
    cmap: str = 'viridis',
    xlabel: str = '',
    ylabel: str = '',
    label: str = 'Magnitude',
    scale: Optional[float] = None,
    return_fig=True,name='',working_dir='.',
) -> None:
    """
    Plot a 2D vector field with arrow lengths and colors controlled by optional scalar fields.

    Parameters
    ----------
    x_coordinate : 1D array, shape (Nx,)
        x-coordinates.
    y_coordinate : 1D array, shape (Ny,)
        y-coordinates.
    Vx : 2D array, shape (Nx, Ny)
        x-component of the vector field.
    Vy : 2D array, shape (Nx, Ny)
        y-component of the vector field.
    A : 2D array, shape (Nx, Ny), optional
        Scalar field determining arrow lengths. Should be non-negative. If None, the vector magnitude is used.
    B : 2D array, shape (Nx, Ny), optional
        Scalar field determining arrow colors. If None, the vector magnitude is used.
    threshold : float, optional
        Minimum vector magnitude to plot an arrow. Vectors with magnitude below this threshold are ignored.
    step_x : int, optional
        Plotting step in the x-direction: only every `step_x`-th point is drawn.
    step_y : int, optional
        Plotting step in the y-direction: only every `step_y`-th point is drawn.
    aspect : Optional[Union[str, float]], optional
        Aspect ratio of the plot. If 'auto', the aspect ratio is determined automatically. If a float, it specifies the aspect ratio.
    cmap : str, optional
        Colormap name for the arrow colors.
    name : str, optional
        Plot title.
    xlabel, ylabel : str, optional
        Axis labels.
    label : str, optional
        Label for the colorbar.
    """
    # Subset data for plotting
    x_slice=slice_between(x_coordinate, xmin, xmax, step=step_x)
    y_slice=slice_between(y_coordinate, ymin, ymax, step=step_y)
    x_coordinate=x_coordinate[x_slice]
    y_coordinate=y_coordinate[y_slice]
    Nx=x_coordinate.size
    Ny=y_coordinate.size
    xmin=x_coordinate[0]
    xmax=x_coordinate[-1]
    ymin=y_coordinate[0]
    ymax=y_coordinate[-1]
    l_x=xmax-xmin
    l_y=ymax-ymin
    
    Vx=np.asarray(Vx)[x_slice, y_slice]
    Vy=np.asarray(Vy)[x_slice, y_slice]
    magnitude = np.linalg.norm([Vx,Vy],axis=0)
    
    if A is None:
        A = magnitude
    else:
        A = np.asarray(A)[x_slice, y_slice]
    assert np.all(np.nan_to_num(A) >= 0), "A must be non-negative to represent arrow lengths"
    Amax=np.nanmax(A)
    if B is None:
        B = magnitude
    else:
        B = np.asarray(B)[x_slice, y_slice]
    # Ensure all arrays have consistent shape
    assert A.shape == (Nx, Ny) and B.shape == (Nx, Ny) and Vx.shape == (Nx, Ny) and Vy.shape == (Nx, Ny), "A and B must have the same shape as Vx and Vy"
    
    
    if Bmin is None:
        Bmin = np.nanmin(B)
    if Bmax is None:
        Bmax = np.nanmax(B)
    
    if ax_dict is None:
        ax_main_height=6   #unit: inch
        if type(aspect)==str:
            aspect=l_x/l_y
        ax_main_width=min(ax_main_height*l_x/(l_y*aspect), ax_main_height*3)   #unit: inch
        ax_dict=generate_side_panel_axes(
            ax_main_height=ax_main_height,ax_main_width=ax_main_width,
            generate_ax_legend=False,
        )
    fig:plt.Figure=ax_dict['fig']
    ax_main:plt.Axes=ax_dict['ax_main']
    ax_cbar_divider:AxesDivider=ax_dict['ax_cbar_divider']
    # Compute vector magnitude as default for A and/or B

    if scale is None:
        lx=xmax-xmin
        scale=25*Amax/lx if lx>0 else 1.0
    print(f"Auto scale for quiver: {scale:.3f}")
    norm=Normalize(vmin=Bmin, vmax=Bmax)

    # Create meshgrid for plotting
    X, Y = np.meshgrid(x_coordinate, y_coordinate, indexing='ij')  # shape (Nx, Ny)

    # Compute unit direction vectors (avoid division by zero)
    # Scale direction vectors by A to get the final arrow components
    mask = magnitude > threshold
    U_dir=np.where(mask,A * Vx/magnitude,np.nan)
    V_dir=np.where(mask,A * Vy/magnitude,np.nan)

    # Quiver plot with color mapping based on B
    q = ax_main.quiver(
        X, Y, U_dir, V_dir, B,
        cmap=cmap,
        norm=norm,
        scale_units='x',
        angles='xy',
        width=0.005,
        headwidth=1,
        headlength=5,
        scale=scale,
    )

    # Set axis properties
    ax_main.set_xlabel(xlabel)
    ax_main.set_ylabel(ylabel)
    ax_main.set_xlim(xmin, xmax)
    ax_main.set_ylim(ymin, ymax)
    
    ax_main.set_aspect('auto')
    ax_main.grid(True, linestyle='--', alpha=0.5)
    fig.suptitle(name, fontsize=25)
    add_colorbar(q, ax_cbar_divider, label=label)


    if return_fig:
        return {'fig': fig, 'ax_main': ax_main, 'ax_cbar_divider': ax_cbar_divider}
    else:
        return savefig(fig=fig, fig_path=os.path.join(working_dir, f'{name}.png'))



def plot_complex_field_2D(
    A:np.ndarray=None, 
    phase:np.ndarray=None,amplitude:np.ndarray=None,
    A_max=None,
    x_axis=None,y_axis=None,
    xlabel='x',ylabel='y',label='',
    xmin:Optional[float]=None,xmax:Optional[float]=None,ymin:Optional[float]=None,ymax:Optional[float]=None,
    plot_polar_colorbar=False,
    return_fig=False,name='',
    working_dir='.'
    ):
    """
    绘制二维复数场，并使用极坐标colorbar同时展示相位和振幅
    All variables should be normalized in advance.
    参数:
    A: 二维复数数组
    """
    assert (A is not None) or (phase is not None and amplitude is not None), "Either A or both phase and amplitude must be provided."
    if A is not None:
        A=np.asarray(A)
        phase = np.angle(A)  # 范围 [-π, π]
        amplitude = np.abs(A)
        Nx,Ny=phase.shape
    if x_axis is None:
        x_axis=np.arange(Nx)
    if y_axis is None:
        y_axis=np.arange(Ny)
    assert phase.shape==(Nx,Ny)
    assert amplitude.shape==(Nx,Ny)
    # 计算相位和振幅
    norm=Normalize(vmin=0, vmax=1)
    rgb_image = phase_amp_to_rgb(phase, amplitude, amplitude_max=A_max)  #shape (Nx, Ny, 3)
    sm = ScalarMappable(norm=norm, cmap='hsv')
    sm.set_array([])
    if xmin is None:
        xmin=x_axis[0]
    if xmax is None:
        xmax=x_axis[-1]
    if ymin is None:
        ymin=y_axis[0]
    if ymax is None:
        ymax=y_axis[-1]
    # 创建图形
    fig,ax_main = plt.subplots()
    pcm = ax_main.pcolormesh(x_axis,y_axis,np.zeros((Ny,Nx)),color=rgb_image.transpose(1,0,2).reshape(-1, 3),shading='auto')
    ax_main.set_aspect('equal')
    ax_main.set_xlabel(xlabel, fontsize=12)
    ax_main.set_ylabel(ylabel, fontsize=12)
    ax_main.set_xlim(xmin,xmax)
    ax_main.set_ylim(ymin,ymax)
    ax_main.set_title(label, fontsize=12)
    ax_main.grid(True, alpha=0.2, linestyle='--')
    ax_cbar=plt.colorbar(mappable=sm,ax=ax_main).ax
    phase_ticks = [0, 0.25, 0.5, 0.75, 1]
    phase_tick_labels = ['-π', '-π/2', '0', 'π/2', 'π']
    ax_cbar.set_yticks(phase_ticks)
    ax_cbar.set_yticklabels(phase_tick_labels, fontsize=9)
    ax_cbar.set_ylabel('phase (rad)', fontsize=10)
    if plot_polar_colorbar:
        plot_polar_hsv_colorbar(label=label)
    if return_fig:
        return {'fig': fig, 'ax_main': ax_main}
    else:
        return savefig(fig=fig, fig_path=os.path.join(working_dir, f'{name}.png'))

    

def plot_polar_hsv_colorbar(label=''):
    Nr=100   #nomber of radial divisions
    Nt=200   #number of angular divisions

    r_axis = np.linspace(0, 1, Nr)  # 半径表示振幅 [0, 1]
    theta_axis = np.linspace(0, 2*np.pi, Nt,endpoint=False)  # 角度表示相位 [0, 2π]
    
    r, theta = np.meshgrid(r_axis, theta_axis,indexing='ij')
    rgb_image = phase_amp_to_rgb(theta, r)  #shape (Nr, Nt, 3)
    fig = plt.figure()
    ax_polar = fig.add_subplot(111, projection='polar')
    
    pcm = ax_polar.pcolormesh(theta_axis, r_axis,np.zeros((Nr,Nt)),shading='auto', color=rgb_image.reshape(-1, 3))
    
    # 设置极坐标图属性
    ax_polar.set_theta_zero_location('E')  # 0度在右侧
    ax_polar.set_theta_direction(1)  # 角度逆时针增加
    ax_polar.set_rlim(0, 1)
    
    # 设置相位刻度 (角度)
    phase_ticks = np.linspace(0, 2*np.pi, 8, endpoint=False)
    phase_labels = ['0°', '45°', '90°', '135°', '180°', '225°', '270°', '315°']
    ax_polar.set_xticks(phase_ticks)
    ax_polar.set_xticklabels(phase_labels, fontsize=9)
    
    # 设置振幅刻度 (半径)
    amp_ticks = [0, 0.25, 0.5, 0.75, 1.0]
    amp_labels = ['0', '0.25', '0.5', '0.75', f'{label}=1.0']
    ax_polar.set_yticks(amp_ticks)
    ax_polar.set_yticklabels(amp_labels, fontsize=9)
    
    # 添加极坐标colorbar标题和标签
    ax_polar.set_title('phase-amplitude', fontsize=12, pad=20)
    
    # 添加参考线
    ax_polar.plot([0, 0], [0, 1], 'w--', alpha=0.5, linewidth=0.8)  # 0°参考线
    ax_polar.plot([np.pi/2, np.pi/2], [0, 1], 'w--', alpha=0.5, linewidth=0.8)  # 90°参考线
    ax_polar.plot([np.pi, np.pi], [0, 1], 'w--', alpha=0.5, linewidth=0.8)  # 180°参考线
    ax_polar.plot([3*np.pi/2, 3*np.pi/2], [0, 1], 'w--', alpha=0.5, linewidth=0.8)  # 270°参考线
    
    # 添加圆形网格
    for r_val in amp_ticks:
        circle = Circle((0, 0), r_val, transform=ax_polar.transData._b, 
                       fill=False, edgecolor='white', alpha=0.3, linewidth=0.5)
        ax_polar.add_artist(circle)
    norm=Normalize(vmin=0, vmax=1)
    sm = ScalarMappable(norm=norm, cmap='hsv')
    sm.set_array([])
    ax_cbar=plt.colorbar(mappable=sm,ax=ax_polar).ax
    phase_ticks = [0, 0.25, 0.5, 0.75, 1]
    phase_tick_labels = ['-π', '-π/2', '0', 'π/2', 'π']
    ax_cbar.set_yticks(phase_ticks)
    ax_cbar.set_yticklabels(phase_tick_labels, fontsize=9)
    ax_cbar.set_ylabel('phase (rad)', fontsize=10)
    savefig(fig=fig,fig_path=os.path.join(working_dir,f'Polar_colorbar_{label}.png'))


if __name__ == "__main__":
    pass
