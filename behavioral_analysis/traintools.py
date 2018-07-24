import numpy as np

from functools import partial

from bokeh.layouts import column, row, layout
from bokeh.models import ColumnDataSource, Slider, Button
from bokeh.themes import Theme
from bokeh.io import show
from bokeh.plotting import figure, output_file, Column
from bokeh.models import DataTable, TableColumn, PointDrawTool, ColumnDataSource, CrosshairTool, CDSView, BooleanFilter
from bokeh.events import DoubleTap
from bokeh.models.widgets import TextInput
from bokeh.palettes import viridis

import segmentation
import bootcamp_utils

def _check_ims(ims):
    if not segmentation._check_array_like(ims):
        raise RuntimeError("The given ims object is not array like, it is " + str(type(ims)))
    [segmentation._check_image_input(im) for im in ims]
    
def point_label(ims, point_size=3, table_height=200, crosshair_tool_alpha=0.5,
                point_tool_color='white'):
        
    _check_ims(ims)
    ims = np.array(ims)
    
    max_height = max(np.array([b.shape for b in ims])[:, 0])
    max_width = max(np.array([b.shape for b in ims])[:, 1])

    point_labels = ColumnDataSource({'x': [], 'y': [], 'frame': []})

    def modify_doc(doc):
        im_num = [0,]

        images = [np.pad(im, ((max_height-im.shape[0], 0), (0, max_width-im.shape[1])), 'constant') for im in ims]

        plot, source = bootcamp_utils.viz.bokeh_imshow(images[im_num[-1]], return_im=True)
        source = source.data_source

        booleans = [True if frame == im_num[-1] else False for frame in point_labels.data['frame']]
        view = CDSView(source=point_labels, filters=[BooleanFilter(booleans)])

        renderer = plot.scatter(x='x', y='y', source=point_labels, view=view,
                                color=point_tool_color, size=point_size)
        columns = [TableColumn(field="x", title="x"),
                   TableColumn(field="y", title="y"),
                   TableColumn(field='frame', title='frame')]
        table = DataTable(source=point_labels, columns=columns, editable=True, height=table_height)
        draw_tool = PointDrawTool(renderers=[renderer], empty_value=im_num[-1])
        plot.add_tools(draw_tool)
        plot.add_tools(CrosshairTool(line_alpha=crosshair_tool_alpha))
        plot.toolbar.active_tap = draw_tool
        
        def update_image(new_ind):
            _, data = bootcamp_utils.viz.bokeh_imshow(images[new_ind], return_im=True)
            data = data.data_source
            source.data = data.data
        
        def callback_point_view(event):
            booleans = [True if frame == im_num[-1] else False for frame in point_labels.data['frame']]
            view = CDSView(source=point_labels, filters=[BooleanFilter(booleans)])
            renderer.view = view

        def callback_slider(attr, old, new):
            update_image(new)
            
            im_num.append(int(new))
            draw_tool.empty_value = im_num[-1]
            callback_point_view('tap')

        def callback_button(direction):
            new = im_num[-1]+direction
            if (((len(images) - 1) < new and direction == 1) or
                (new == -1 and direction == -1)):
                return None
            update_image(new)
            
            im_num.append(new)
            draw_tool.empty_value = im_num[-1]
            callback_point_view('tap')
            
        slider = Slider(start=0, end=len(images), value=0, step=1, title="Frame Number")
        slider.on_change('value', callback_slider)
        
        button_back = Button(label='back',button_type="success")
        button_back.on_click(partial(callback_button, direction=-1))

        button_forward = Button(label='forward',button_type="success")
        button_forward.on_click(partial(callback_button, direction=1))
        
        plot.on_event('tap', callback_point_view)

        doc.add_root(column(row(slider), plot, row(button_back, button_forward), table))

    show(modify_doc)
    return point_labels

def button_label(ims, button_values=('beetle', 'ant')):
    _check_ims(ims)
    ims = np.array(ims)
    
    max_height = max(np.array([b.shape for b in ims])[:, 0])
    max_width = max(np.array([b.shape for b in ims])[:, 1])
    
    frame_labels = ColumnDataSource({'type': [],
                                     'frame': []})

    def modify_doc(doc):
        im_num = [0,]
    
        images = [np.pad(im, ((max_height-im.shape[0],0), (0, max_width-im.shape[1])), 'constant') for im in ims]

        plot, source = bootcamp_utils.viz.bokeh_imshow(images[im_num[-1]], return_im=True)
        source = source.data_source

        columns = [TableColumn(field='type', title='type'),
                   TableColumn(field='frame', title='frame')]
        table = DataTable(source=frame_labels, columns=columns, editable=True, height=200)
        plot.add_tools(CrosshairTool(line_alpha=0.5))

        def callback(attr, old, new):
            im_num.append(int(new))
            temp_plot, data = bootcamp_utils.viz.bokeh_imshow(images[int(new)], return_im=True)
            data = data.data_source
            source.data = data.data
            plot.x_range.end = temp_plot.x_range.end
            #plot.plot_width = temp_plot.plot_width
            #layout.children[1] = plot

        def callback_button(direction):
            if (((len(images) - 2) < im_num[-1] and direction == 1) or
                (im_num[-1] == 0 and direction == -1)):
                return None
            _, data = bootcamp_utils.viz.bokeh_imshow(images[im_num[-1]+direction], return_im=True)
            im_num.append(im_num[-1]+direction)
            data = data.data_source
            source.data = data.data
            draw_tool.empty_value = im_num[-1]
            callback_point_view('tap')

        def callback_label_button(value):        
            new_data = {'type': [value],
                        'frame': [im_num[-1]]}
            frame_labels.stream(new_data)

            if (len(images) - 2) < im_num[-1]:
                return None
            _, data = bootcamp_utils.viz.bokeh_imshow(images[im_num[-1]+1], return_im=True)
            im_num.append(im_num[-1]+1)
            data = data.data_source
            source.data = data.data

        slider = Slider(start=0, end=len(images)-1, value=0, step=1, title="Frame Number")
        slider.on_change('value', callback)
        
        button_back = Button(label='back',button_type="success")
        button_back.on_click(partial(callback_button, direction=-1))

        button_forward = Button(label='forward',button_type="success")
        button_forward.on_click(partial(callback_button, direction=1))

        label_buttons = [Button(label=value, button_type='success') for value in button_values]
        [button.on_click(partial(callback_label_button, value=value)) for button, value in zip(label_buttons, button_values)]

        #for a grid layout of the buttons, we need to pad the list with an empty spot if the button count is not even
        if not np.isclose(len(label_buttons) % 2, 0):
            label_buttons.append(Button(label=''))
        buttons = np.reshape(label_buttons, (-1,2))
        buttons = buttons.tolist()

        layout_list = [[slider], [plot],
                       [button_back, button_forward]]
        [layout_list.append(button) for button in buttons]
        layout_list.append([table])

        doc.add_root(layout(layout_list))

    show(modify_doc)
    return frame_labels