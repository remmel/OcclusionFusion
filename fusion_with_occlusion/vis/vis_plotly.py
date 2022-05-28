import os
import json 
from .visualizer import Visualizer

from plotly.subplots import make_subplots
import plotly.offline as pyo
import plotly.graph_objs as go

class VisualizerPlotly(Visualizer):
	def __init__(self,opt):
		super().__init__(opt)	


	def plot_motion_complete_tests_rec_error(self):
		datadir = self.opt.datadir
		rec_err = {}
		for i in range(2,6):
			filepath = os.path.join(datadir,"results",f"occ_fusion_test{i}.txt")
			with open(filepath) as f:
				data = [ float(x) for x in f.read().split() if len(x) > 0]
			rec_err[f"test_{i}"] = np.array(data)

		x = list(range(len(rec_err["test_2"])))

		fig = make_subplots(rows=1, cols=1, subplot_titles=[f"Graph Nodes Reconstructon Error"])
		for i in range(2,6):

			if i == 2: 
				test_name = "Test:1 Ground Truth Source Nodes"
			elif i == 3:
				test_name = "Test:2 Predicted Source Nodes"
			elif i == 4: 
				test_name = "Test:3 Predicted Source Nodes + ARAP Loss"
			elif i == 5:
				test_name = "Test:4 Predicted Source Nodes + ARAP & Data Loss"
				
			fig.add_trace(go.Scatter(x = x,
									 y = rec_err[f"test_{i}"],
									 mode="lines + text",
									 name=test_name),row=1,col=1)
					 
		fig.update_layout(xaxis_title="Frame Index", yaxis_title="Rec Err.",font=dict(size=25))
		fig.show()


	def plot_convergance_info(self):

		sample_name = os.path.basename(self.opt.datadir)
		dir_path = os.path.join(self.opt.datadir,"results")


		files = sorted([ x for x in os.listdir(dir_path) if "optimization_convergence_info" in x],key=lambda x: int(x.split('_')[-1].split('.')[0]))

		loss_terms = {'target_id':[],'data':[],'total':[],'arap':[],'3D':[],'px':[],'py':[],'motion':[]}

		for file in files: 
			target_id = int(file.split('_')[-1].split('.')[0])
			with open(os.path.join(dir_path,file)) as f:
				convergance_info = json.load(f)

			for term in loss_terms: 
				if term == 'target_id':
					loss_terms[term].append(target_id)
				else:     
					if term not in convergance_info:
						loss_terms[term].append(0)
					else: 
						loss_terms[term].append(convergance_info[term][-1])

		fig = make_subplots(rows=1, cols=1, subplot_titles=[f"Loss terms during Optimization for:{sample_name}"])

		for term in loss_terms:
			if term == "target_id": 
				continue

			fig.add_trace(go.Scatter(x = loss_terms["target_id"],
									 y = loss_terms[term],
									 mode="lines + text",
									 name=term),row=1,col=1)

		if "donkeydoll" in sample_name.lower():	
			fig.add_vrect(x0=0, x1=9, line_width=0, fillcolor="green", opacity=0.2,annotation_text="No<br>Movement", annotation_position="top left")
			fig.add_vrect(x0=9, x1=48, line_width=0, fillcolor="red", opacity=0.2,annotation_text="Arms moving forward", annotation_position="top left")
			fig.add_vrect(x0=48, x1=79, line_width=0, fillcolor="green", opacity=0.2,annotation_text="Arms moving backwards", annotation_position="top left")
			fig.add_vrect(x0=79, x1=119, line_width=0, fillcolor="red", opacity=0.2,annotation_text="Head moving down", annotation_position="top left")
			fig.add_vrect(x0=119, x1=146, line_width=0, fillcolor="green", opacity=0.2,annotation_text="Head moving up", annotation_position="top left")
			fig.add_vrect(x0=146, x1=154, line_width=0, fillcolor="red", opacity=0.2,annotation_text="No<br>Movement", annotation_position="top left")
		
		if "seq002" in sample_name.lower():		
			fig.add_vrect(x0=0, x1=27, line_width=0, fillcolor="green", opacity=0.2,annotation_text="No<br>Movement", annotation_position="top left")
			fig.add_vrect(x0=27, x1=loss_terms["target_id"][-1], line_width=0, fillcolor="red", opacity=0.2,annotation_text="T-shirt is lowered", annotation_position="top left")
					 
		fig.update_layout(xaxis_title="Target Frame Index", yaxis_title="Value.",font=dict(size=25))		
		fig.show()
		fig.write_html(os.path.join(dir_path,"convergance_info_plot.html"))