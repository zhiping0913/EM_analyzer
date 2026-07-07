import numpy as np
import matplotlib.pyplot as plt
from matplotlib.axes import Axes
import os

from typing import Optional,List,Dict
from plot.plot_basic import savefig


def plot_multiple_1D_fields(
    coordinate:np.ndarray,
    field_dict_list:Optional[List[Dict]]=None,
    axhline_dict_list:Optional[List[Dict]]=None,
    axvline_dict_list:Optional[List[Dict]]=None,
    ax: Optional[Axes]=None,
    xmin:Optional[float]=None,xmax:Optional[float]=None,
    ymin:Optional[float]=None,ymax:Optional[float]=None,
    ylabel=r'$a=\frac{E}{E_c}=\frac{B}{B_c}$',xlabel=r'$\frac{x}{\lambda_0}$',
    xscale='linear',yscale='linear',
    coordinate_direction='vertical',
    plot_legend:bool=True,set_label=True,
    name='',working_dir='.',return_fig:bool=True
    ):
    r"""_summary_

    Args:
        field_dict_list=[{'field':field,'linestyle':'-','label':None,'color':None,'linewidth':2},...] 
        axhline_dict_list=[{'y':y,'linestyle':'--','label':None,'color':None,'alpha':0.5,'linewidth':2},...]
        axvline_dict_list=[{'x':x,'linestyle':'--','label':None,'color':None,'alpha':0.5,'linewidth':2},...]
        coordinate (np.ndarray): _description_
        ax (Optional[Axes], optional): _description_. Defaults to None.
        xmin (Optional[float], optional): _description_. Defaults to None.
        xmax (Optional[float], optional): _description_. Defaults to None.
        ymin (Optional[float], optional): _description_. Defaults to None.
        ymax (Optional[float], optional): _description_. Defaults to None.
        ylabel (str, optional): _description_. Defaults to r'$a=\frac{E}{E_c}=\frac{B}{B_c}$'.
        xlabel (str, optional): _description_. Defaults to r'$\frac{x}{\lambda_0}$'.
        coordinate_direction (str, optional): The direction of the plot. Defaults to 'vertical'. 'x' or 'vertical' for vertical plot where coordinate is in x-direction, 'y' or 'horizontal' for horizontal plot,where coordinate is in y-direction.
        name (str, optional): _description_. Defaults to ''.
        working_dir (str, optional): _description_. Defaults to '.'.
        return_fig (bool, optional): _description_. Defaults to True.
    """
    assert coordinate_direction in ['x','y','vertical','horizontal'], 'direction must be either "x", "y", "vertical" or "horizontal".'
    if ax is None:
        fig, ax = plt.subplots(figsize=(8,6))
    else:
        fig = ax.figure
    ax.set_xscale(xscale)
    ax.set_yscale(yscale)
    ax.grid(True, alpha=0.3)
    if field_dict_list is not None:
        for field_dict in field_dict_list:
            field = field_dict['field']
            assert len(coordinate)==len(field), f'The length of coordinate and field must be the same. While len(coordinate)={len(coordinate)}, len(field)={len(field)}'
            linestyle = field_dict.get('linestyle','-')
            label = field_dict.get('label',None)
            color = field_dict.get('color',None)
            linewidth = field_dict.get('linewidth',2)
            if coordinate_direction in ['x','vertical']:
                ax.plot(coordinate,field,linestyle=linestyle,label=label,color=color,linewidth=linewidth)
            elif coordinate_direction in ['y','horizontal']:
                ax.plot(field,coordinate,linestyle=linestyle,label=label,color=color,linewidth=linewidth)
    if axhline_dict_list is not None:
        for axhline_dict in axhline_dict_list:
            y = axhline_dict['y']
            linestyle = axhline_dict.get('linestyle','--')
            label = axhline_dict.get('label',None)
            color = axhline_dict.get('color',None)
            alpha = axhline_dict.get('alpha',0.5)
            linewidth = axhline_dict.get('linewidth',2)
            ax.axhline(y=y,linestyle=linestyle,label=label,color=color,alpha=alpha,linewidth=linewidth)
    if axvline_dict_list is not None:
        for axvline_dict in axvline_dict_list:
            x = axvline_dict['x']
            linestyle = axvline_dict.get('linestyle','--')
            label = axvline_dict.get('label',None)
            color = axvline_dict.get('color',None)
            alpha = axvline_dict.get('alpha',0.5)
            linewidth = axvline_dict.get('linewidth',2)
            ax.axvline(x=x,linestyle=linestyle,label=label,color=color,alpha=alpha,linewidth=linewidth)
    if set_label:
        ax.set_xlabel(xlabel,fontsize=20)
        ax.set_ylabel(ylabel,fontsize=20)
    ax.set_title(name,fontsize=20)
    if xmin is not None and xmax is not None:
        ax.set_xlim(xmin,xmax)
    if ymin is not None and ymax is not None:
        ax.set_ylim(ymin,ymax)
    if plot_legend:
        ax.legend(fontsize=14, loc='upper left', bbox_to_anchor=(1.02, 1.0), borderaxespad=0.0)   #move legend outside of the plot
    if return_fig:
        ax_dict={
            'fig':fig,
            'ax_main':ax,
        }
        return ax_dict
    else:
        plt.tight_layout()
        return savefig(fig,fig_path=os.path.join(working_dir,name+'.png'))

def plot_twinx(
    coordinate:np.ndarray,
    field_dict_list_1:list[dict] ,
    field_dict_list_2:list[dict] ,
    xmin:Optional[float]=None,
    xmax:Optional[float]=None,
    xlabel:Optional[str]="",
    y1_label:Optional[str]="",y1_scale:Optional[str]="linear",y1_min:Optional[float]=None,y1_max:Optional[float]=None,
    y2_label:Optional[str]="",y2_scale:Optional[str]="linear",y2_min:Optional[float]=None,y2_max:Optional[float]=None,
    color_1:Optional[str]="red",color_2:Optional[str]="blue",
    ax:Optional[Axes]=None,
    return_ax:Optional[bool]=True,name:Optional[str]="",working_dir:Optional[str]='.'
):
    """_summary_

    Args:
        coordinate (_type_): _description_
        field_dict_list_1,field_dict_list_2 (_type_): 
            field_dict_list=[{'field':field,'linestyle':'-','label':None,'color':color_1 or color_2},...] 
        y1 (_type_): _description_
        y2 (_type_): _description_
        xmin (_type_, optional): _description_. Defaults to None.
        xmax (_type_, optional): _description_. Defaults to None.
        xlabel (str, optional): _description_. Defaults to "".
        y1_label (str, optional): _description_. Defaults to "".
        y1_scale (str, optional): _description_. Defaults to "linear".
        y1_min (_type_, optional): _description_. Defaults to None.
        y1_max (_type_, optional): _description_. Defaults to None.
        y2_label (str, optional): _description_. Defaults to "".
        y2_scale (str, optional): _description_. Defaults to "linear".
        y2_min (_type_, optional): _description_. Defaults to None.
        y2_max (_type_, optional): _description_. Defaults to None.
        ax (Optional[Axes], optional): _description_. Defaults to None.
        return_ax (bool, optional): _description_. Defaults to True.
        name (str, optional): _description_. Defaults to "".
        working_dir (str, optional): _description_. Defaults to '.'.

    Returns:
        _type_: _description_
    """
    if xmin is None:
        xmin = np.min(coordinate)
    if xmax is None:
        xmax = np.max(coordinate)
    if ax is None:
        fig = plt.figure(figsize=(8, 6))
        ax1: Axes = fig.add_axes([0.1, 0.1, 0.8, 0.8])
        ax2: Axes = ax1.twinx()
    else:
        fig = ax.get_figure()
        ax1 = ax
        ax2 = ax1.twinx()
    line_list=[]
    for field_dict in field_dict_list_1:
        field=field_dict['field']
        linestyle=field_dict.get('linestyle','-')
        label=field_dict.get('label',None)
        color=field_dict.get('color',color_1)
        line,=ax1.plot(coordinate, field, linestyle=linestyle, color=color, label=label)
        line_list.append(line)
    ax1.set_ylim(y1_min, y1_max)
    ax1.set_xlabel(xlabel)
    ax1.set_ylabel(y1_label, color=color_1)
    ax1.set_yscale(y1_scale)
    ax1.tick_params(axis="y", labelcolor=color_1,left=True,right=False,labelleft=True,labelright=False)
    ax1.set_xlim(xmin, xmax)
    for field_dict in field_dict_list_2:
        field=field_dict['field']
        linestyle=field_dict.get('linestyle','-')
        label=field_dict.get('label',None)
        color=field_dict.get('color',color_2)
        line,=ax2.plot(coordinate, field, linestyle=linestyle, color=color, label=label)
        line_list.append(line)
    ax2.set_ylim(y2_min, y2_max)
    ax2.set_ylabel(y2_label, color=color_2)  # we already handled the x-label with ax1
    ax2.set_yscale(y2_scale)
    ax2.tick_params(axis="y", labelcolor=color_2,left=False,right=True,labelleft=False,labelright=True)
    ax1.grid(True, which='both', axis='both', linestyle='--', alpha=0.7)
    ax2.set_xlim(xmin, xmax)
    fig.legend(handles=line_list, loc='upper right')
    fig.suptitle(name)
    fig.tight_layout()  # otherwise the right y-label is slightly clipped
    if return_ax:
        return {"fig": fig, "ax1": ax1, "ax2": ax2}
    else:
        return savefig(fig,fig_path=os.path.join(working_dir,f'{name}.png' ))


def plot_family_1D_fields(
    x_coordinate:np.ndarray, t_coordinate:np.ndarray,
    field_dict_list:List[Dict],
    x_label:str='x', t_label:str='t', label:str='f(t,x)',
    xmin:Optional[float]=None, xmax:Optional[float]=None, tmin:Optional[float]=None, tmax:Optional[float]=None,ymin:Optional[float]=None, ymax:Optional[float]=None,
    plot_legend:bool=True,plot_cbar:bool=True,
    ax:Optional[plt.Axes]=None,ax_cbar:Optional[plt.Axes]=None,
    name='',working_dir='.',return_fig:bool=True
    
    ):
    """_summary_

    Args:
        x_coordinate (np.ndarray): _description_
        t_coordinate (np.ndarray): _description_
        field_dict_list=[{'field':field,'linestyle':'-','label':None,'color':None},...] 

    Returns:
        None
    """
    x_coordinate=np.asarray(x_coordinate).flatten()
    t_coordinate=np.asarray(t_coordinate).flatten()
    Nx=x_coordinate.size
    Nt=t_coordinate.size
    assert len(field_dict_list)==Nt, "Length of t_coordinate must match length of field_dict_list"
    cmap = plt.get_cmap('tab10')
    norm = plt.Normalize(vmin=np.min(t_coordinate), vmax=np.max(t_coordinate))
    
    # Create 3D figure
    if ax is None:
        fig = plt.figure(figsize=(10, 7))
        ax = fig.add_subplot(111, projection='3d')
        ax_cbar=fig.add_axes([0.85, 0.15, 0.02, 0.7]) if plot_cbar else None
    else:
        fig=ax.get_figure()
        if ax_cbar is None:
            ax_cbar=ax

    # Plot each curve as a 3D line with color based on t
    for t, field_dict in zip(t_coordinate, field_dict_list):
        field=field_dict['field']
        assert len(field)==Nx, "Length of field must match length of x_coordinate"
        ax.plot(x_coordinate, field, [t] * Nx,    # x, y, z = (x, f, t)
                color=field_dict.get('color',cmap(norm(t))), 
                linestyle=field_dict.get('linestyle','-'), 
                linewidth=2, 
                label=field_dict.get('label',f't = {t}'),
                )

    sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
    sm.set_array([])
    if plot_cbar:
        cbar = fig.colorbar(sm, cax=ax_cbar)
        cbar.set_label(t_label)

    ax.set_xlabel(x_label)
    ax.set_ylabel(label)   # vertical axis
    ax.set_zlabel(t_label)          # depth axis
    ax.set_xlim(xmin, xmax)
    ax.set_ylim(ymin, ymax)
    ax.set_zlim(tmin, tmax)
    # Ensure y‑axis (f) is vertical and adjust viewing angle
    ax.view_init(elev=20, azim=-60, vertical_axis='y')
    ax.set_title(name)

    # Optional legend (though colorbar already encodes t)
    if plot_legend:
        ax.legend()

    plt.tight_layout()
    if return_fig:
        return {'fig': fig, 'ax': ax,}
    else:
        return savefig(fig,fig_path=os.path.join(working_dir,f'{name}.png' ))
