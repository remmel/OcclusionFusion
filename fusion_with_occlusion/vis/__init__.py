def get_visualizer(opt):
	

	if opt.visualizer.lower() == "matplotlib": 
		from .vis_matplotlib import VisualizeMatplotlib
		return VisualizeMatplotlib(opt)

	elif opt.visualizer.lower() == "open3d": 
		from .vis_open3d import VisualizeOpen3D
		return VisualizeOpen3D(opt)

	elif opt.visualizer.lower() == "plotly": 
		from .vis_plotly import VisualizerPlotly
		return VisualizerPlotly(opt)
	elif opt.visualizer.lower() == "ipyvolume":
		from .vis_ipyvolume import VisualizerIpyVolume
		return VisualizerIpyVolume(opt)
	else: 
		print("Reaching here")
		raise NotImplementedError("Current possible visualizers -> matplotlib,open3d,plotly. eg. --visualizers open3d")


