import _init_paths
import tensorflow as tf
from fast_rcnn.config import cfg
from fast_rcnn.test import im_detect
from fast_rcnn.nms_wrapper import nms
from utils.timer import Timer
import matplotlib.pyplot as plt
import numpy as np
import os, sys, cv2
import argparse
from networks.factory import get_network
from xml.etree.ElementTree import Element, SubElement, dump, ElementTree, parse

#CLASSES = ('__background__', 'LEFTWD', 'LEFTSD', 'RIGHTWD', 'RIGHTSD')
#CLASSES = ('__background__', 'DEFECT')
CLASSES = ('__background__', 'SD', 'WD')

def vis_detections(im, class_name, dets, ax, thresh=0.5):
	"""
	visualize predicted boxes
	"""
	if class_name == 'SD':
		color_ = 'red'
		color_str = 'green'
	else:
		color_ = 'blue'
		color_str = 'white'
	
	"""Draw detected bounding boxes."""
	inds = np.where(dets[:, -1] >= thresh)[0]
	if len(inds) == 0:
		plt.tight_layout()
		plt.draw()
		return

	for i in inds:
		bbox = dets[i, :4]
		score = dets[i, -1]
		
		ax.add_patch(
			plt.Rectangle((bbox[0], bbox[1]),
						  bbox[2] - bbox[0],
						  bbox[3] - bbox[1], fill=False,
						  edgecolor=color_, linewidth=2)
			)
		ax.text(bbox[0], bbox[1] - 2,
				'{:s} {:.3f}'.format(class_name, score),
				bbox=dict(facecolor=color_, alpha=0.5),
				fontsize=7, color='white')

	#ax.set_title(('detections with '
	#			   'p(d|box) >= 0.3 , ' 'p(n|box) >= 0.7' '\n \n').format(class_name, class_name, thresh),
	#			   fontsize=25)
	
	#plt.axis('off')
	#plt.tight_layout()
	plt.draw()

# defined by Hyoje	  
# calculate the IOU between 'SD' and 'WD'
#
def cal_IOU(rec_fix, rects):
	"""
	calcuate IOU.	
	"""
	max_ = np.maximum(rec_fix, rects)
	min_ = np.minimum(rec_fix, rects)

	xlen = min_[:, 2] - max_[:, 0]
	ylen = min_[:, 3] - max_[:, 1]
	idx = np.where((xlen > 0) & (ylen > 0))[0]
	nonoverlab_idx = np.where(~((xlen > 0) & (ylen > 0)))[0]
	intersection = (xlen*ylen)[idx]

	org1 = (rects[idx, 2] - rects[idx, 0]) * (rects[idx, 3] - rects[idx, 1])
	org2 = (rec_fix[2] - rec_fix[0]) * (rec_fix[3] - rec_fix[1])

	IOU = intersection / (org1 + org2 - intersection)
	
	return IOU, idx, nonoverlab_idx

# defined by Hyoje	  
# get roi removed overlaped things
def get_roi(sess, net, im, CONF_THRESH):
	"""
	get rois
	return a prediction ROI from splited image.
	
	args : 
		sess : tf.Session()
		net : networks(faster r cnn)
		im (ndarray) : image
		CONF_THRESH (list) : thersholds of objects
	
	return :
		pred_roi (dict) : 'object_name' (ndarray) : coordinates of bounding boxes and score	
	"""
	timer = Timer()
	timer.tic()
	# Detect all object classes and regress object bounds
	scores, boxes = im_detect(sess, net, im)	# 2000 proposal, scores = (2000, 3), boxes = (2000, 12)
	
	#inds = np.where(scores[:, 1] < CONF_THRESH[1])[0]
	#scores[inds, 2] += scores[inds, 1]
	
	#scores[:, 2] += scores[:, 1]
	timer.toc()
	print ('Detection took {:.3f}s for '
		   '{:d} object proposals').format(timer.total_time, boxes.shape[0])
	# Visualize detections for each class
	im = im[:, :, (2, 1, 0)]

	NMS_THRESH = 0.001
	roi = {}
	for cls_ind, cls in enumerate(CLASSES[1:]):
		cls_ind += 1 # because we skipped background
		cls_boxes = boxes[:, 4*cls_ind:4*(cls_ind + 1)] # shape = (2000, 4)
		cls_scores = scores[:, cls_ind]					# shape = (2000, 1)
		dets = np.hstack((cls_boxes,
						  cls_scores[:, np.newaxis])).astype(np.float32)	#shape = (2000, 5)
		keep = nms(dets, NMS_THRESH)		# remove overlapped boxes
		dets = dets[keep, :]
		inds = np.where(dets[:, -1] >= CONF_THRESH[cls_ind])[0]
		roi[cls] = dets[inds, :]
		
	return roi

def load_xml(xml_name):
	"""
	
	load and get coordinates from xml files
	
	"""
	tree = parse(xml_name)
	root = tree.getroot()

	xmins = []
	ymins = []
	xmaxs = []
	ymaxs = []
	class_names = []

	for obj in root.iter("object"):
		for cls_name in obj.iter("name"):
			class_names.append(cls_name.text)
		for bndbox in obj.iter("bndbox"):
			for val in bndbox.iter("xmin"):
				xmins.append(int(val.text))
			for val in bndbox.iter("ymin"):
				ymins.append(int(val.text))
			for val in bndbox.iter("xmax"):
				xmaxs.append(int(val.text))
			for val in bndbox.iter("ymax"):
				ymaxs.append(int(val.text))
	return xmins, ymins, xmaxs, ymaxs, class_names

def cal_num(pred_roi, true_roi, count):
	"""
	
	calculate the number of results
	the number of predicted object over labeled object
	the number of labeled object over predicted object, etc.
	
	"""
	# stack total prediction of defects, label of defects
	total_pred = []
	for cls in CLASSES[1:]:
		if len(pred_roi[cls]) == 0:
			continue
		else:
			for i in range(len(pred_roi[cls])):
				total_pred.append(pred_roi[cls][i])
	total_pred = np.array(total_pred)

	total_label = []
	for cls in CLASSES[1:]:
		if len(true_roi[cls]) == 0:
			continue
		else:
			for i in range(len(true_roi[cls])):
				total_label.append(true_roi[cls][i])
	total_label = np.array(total_label)

	# the number of prediction(SD or WD) of SD_true
	for i in range(len(true_roi['SD'])):
		if len(total_pred) == 0:
			break
		else:
			iou, _, _ = cal_IOU(true_roi['SD'][i], total_pred)
			if len(iou) == 0:
				continue
			elif len(iou) > 0:
				count['SD_pred_total'] += 1
	# the number of true(SD or WD) of SD_prediction
	for cls in CLASSES[1:]:
		for i in range(len(pred_roi[cls])):
			if len(total_label) == 0:
				break
			else:
				iou, _, _ = cal_IOU(pred_roi[cls][i], total_label)
				if len(iou) == 0:
					continue
				elif len(iou) > 0:
					count['num_ans_of_pred_'+cls] += 1

	for i in range(len(true_roi['SD_pred'])):
		if true_roi['SD_pred'][i] == None:
			continue
		else:
			count['num_ans_of_true_SD'] += 1
	
	return count
	
def vis_test(xml_path, pred_roi, ax):
	"""
	
	visualize labeled boxes and predicted boxes together on demo image
	
	"""
	xmins, ymins, xmaxs, ymaxs, class_names = load_xml(xml_path)

	true_roi = {'SD': [], 'WD':[]}
	for idx, cls in enumerate(class_names):
		temp_bd = [xmins[idx], ymins[idx], xmaxs[idx], ymaxs[idx], 0]
		true_roi[cls.upper()].append(temp_bd)

	for cls in CLASSES[1:]:
		true_roi[cls] = np.array(true_roi[cls]).astype(np.float32)
		true_roi[cls+'_pred'] = []
		pred_roi[cls+'_ans'] = []
		
		if len(pred_roi[cls]) == 0:
			continue			
		for i in range(len(true_roi[cls])):
			iou, pred_idx, _ = cal_IOU(true_roi[cls][i], pred_roi[cls])
			if len(iou) > 1:
				iou = np.array([np.max(iou)])
			elif len(iou) == 0:
				true_roi[cls+'_pred'].append(None)					  
				continue
			true_roi[cls+'_pred'].append(iou[0])
				
		if len(true_roi[cls]) == 0:
			continue
		for i in range(len(pred_roi[cls])):
			iou, ans_idx, _ = cal_IOU(pred_roi[cls][i], true_roi[cls])
			if len(iou) > 1:
				iou = np.array([np.max(iou)])
			elif len(iou) == 0:
				pred_roi[cls+'_ans'].append(None)					 
				continue
			pred_roi[cls+'_ans'].append(iou[0])
			
	count = {'num_pred_SD':0, 'num_pred_WD':0, 'num_label_SD':0, 'num_label_WD':0,
			 'num_ans_of_pred_SD':0, 'num_ans_of_pred_WD':0, 
			 'num_ans_of_true_SD':0, 'SD_pred_total' : 0}
	
	# visualize label defects
	for cls in CLASSES[1:]:
		if cls == 'SD':		   
			_color = 'green'
			_color_str = 'black'
		else:
			_color = 'orange'
			_color_str = 'black'
			
		if len(true_roi[cls+'_pred']) == 0:
			continue
		for i in range(len(true_roi[cls])):			   
			ax.add_patch(plt.Rectangle((true_roi[cls][i][0], true_roi[cls][i][1]),
									  true_roi[cls][i][2] - true_roi[cls][i][0],
									  true_roi[cls][i][3] - true_roi[cls][i][1],
									  fill=False, edgecolor=_color, linewidth=2))
			ax.text(true_roi[cls][i][2], true_roi[cls][i][3] - 2,
				   'Label-{:s}'.format(cls),
				   bbox=dict(facecolor=_color, alpha=0.5),
				   fontsize=8, color=_color_str)
				   
	for cls in CLASSES[1:]:
		count['num_pred_'+cls] = len(pred_roi[cls])
		count['num_label_'+cls] = len(true_roi[cls])				   
	
	count = cal_num(pred_roi, true_roi, count)
	
	#for nm in sorted(count.keys()):
	#	 print('The number of {:s} : {}'.format(nm, count[nm]))
	
	return count
	
def split_4_merge(sess, net, image_name, img_path, CONF_THRESH):
	"""
	first, split imags by 4 and implement faster r cnn.
	return a prediction ROI(merge).
	
	args : 
		sess : tf.Session()
		net : networks(faster r cnn)
		image_name (str) : name of image, S10_M0365-02MS.tif...
		img_path (str) : path of image
		CONF_THRESH (list) : thersholds of objects
	
	return :
		pred_roi (dict) : 'object_name' (ndarray) : coordinates of bounding boxes and score	
	"""
	# image_name : 'S10_M0365-02MS.tif'
	
	# Load the demo image
	# im_file : './data/demo/S10_M0365-02MS.tif'
	im_file = os.path.join(cfg.DATA_DIR, 'demo', image_name) 
	im = cv2.imread(im_file)
	
	n, _, _ = np.shape(im)
	n = n / 2
	overlap = 30
	
	#split image to 4 equal subimages
	img = []
	
	img.append(im[0:n+overlap, 0:n])
	img.append(im[0:n+overlap, n:2*n])
	img.append(im[n-overlap:2*n, 0:n])
	img.append(im[n-overlap:2*n, n:2*n])
		
	
	# helper to calculate coordinates on merged image
	cal_coord_x = [0, n, 0, n, 0]
	cal_coord_y = [0, 0, n-overlap, n-overlap, 0]
	
	# get roi of each objects
	stack_roi = {'SD' : [], 'WD': []}
	for i in range(4):
		temp_roi = get_roi(sess, net, img[i], CONF_THRESH)
		for j in range(len(temp_roi['SD'])):
			if len(temp_roi['WD']) == 0:
				break
			# eliminate the overlapped images
			IOU, _, nonOver_det = cal_IOU(temp_roi['SD'][j], temp_roi['WD'])
			temp_roi['WD'] = temp_roi['WD'][nonOver_det]
			
		for cls in CLASSES[1:]:
			if len(temp_roi[cls]) == 0:
				continue
			temp_coor = [cal_coord_x[i], cal_coord_y[i], cal_coord_x[i], cal_coord_y[i], 0]
			stack_roi[cls].append(temp_roi[cls] + np.array(temp_coor).astype(np.float32))
	
	temp_list = {'SD' : [], 'WD': []}
	pred_roi = {'SD' : [], 'WD': []}
	
	for cls in CLASSES[1:]:
		for i in range(len(stack_roi[cls])):
			for j in range(len(stack_roi[cls][i])):
				temp_list[cls].append(stack_roi[cls][i][j])
		pred_roi[cls] = np.array(temp_list[cls])
	
	  
	fig, ax = plt.subplots(figsize=(12, 12))
	ax.imshow(im, aspect='equal')
	NMS_THRESH = 0.001
	
	# thresholding rois by given thersholds and visualize the rois.
	for idx, cls in enumerate(CLASSES[1:]):
		if len(pred_roi[cls]) == 0:
			continue
		keep = nms(pred_roi[cls], NMS_THRESH)
		pred_roi[cls] = pred_roi[cls][keep, :]		  
		vis_detections(im, cls, pred_roi[cls], ax, thresh=CONF_THRESH[idx+1])
	# S10_M0650-02MS.xml case
	xml_path = os.path.splitext(im_file)[0] + '.xml'
	# 650.xml case
	only_num_xml = os.path.split(im_file)[1]
	only_num_xml = os.path.splitext(only_num_xml)[0]
	only_num_xml = only_num_xml[6:9]
	only_num_xml = only_num_xml + '.xml'
	xml_path2 = os.path.join(img_path, only_num_xml)
	
	count = []
	if os.path.exists(xml_path):	   
		count = vis_test(xml_path, pred_roi, ax)
	elif os.path.exists(xml_path2):
		count = vis_test(xml_path2, pred_roi, ax)
	ax.set_title('Threshold SD : {}	 WD : {}'.format(CONF_THRESH[1], CONF_THRESH[2]), fontsize=24)
	
	plt.tight_layout()
	plt.axis('off')
	
	# save images
	filename = os.path.splitext(os.path.basename(image_name))[0]
	# 'Images/1234.jpg' -> '1234'
	save_path = os.path.join(cfg.DATA_DIR, 'demo/result')
	if not os.path.exists(save_path):
		os.makedirs(save_path)		  
	plt.savefig(save_path+'/demo_{}.jpg'.format(filename))
	
	return count, pred_roi
	

def parse_args():
	"""Parse input arguments."""
	parser = argparse.ArgumentParser(description='Faster R-CNN demo')
	parser.add_argument('--gpu', dest='gpu_id', help='GPU device id to use [0]',
						default=0, type=int)
	parser.add_argument('--cpu', dest='cpu_mode',
						help='Use CPU mode (overrides --gpu)',
						action='store_true')
	parser.add_argument('--net', dest='demo_net', help='Network to use [vgg16]',
						default='VGGnet_test')
	parser.add_argument('--model', dest='model', help='Model path', default=' ')
	parser.add_argument('-p', '--path', dest='demo_path', help='Path to demonstrate data.',
						default='./data/demo/')
	parser.add_argument('-s', '--show', type=int, dest='Do_show', help='Show plt', default=0)
	
	parser.add_argument('-thS', '--thresholdSD', type=float, dest='threshSD', help='threshold', default=0.15)
	parser.add_argument('-thW', '--thresholdWD', type=float, dest='threshWD', help='threshold', default=0.9)
	
	args = parser.parse_args()

	if not args.demo_path:	 # if filename is not given
		parser.error('Error: path to test data must be specified. Pass --path to command line')

	return args	   
   
if __name__ == '__main__':
	cfg.TEST.HAS_RPN = True	 # Use RPN for proposals

	args = parse_args()

	if args.model == ' ':
		raise IOError(('Error: Model not found.\n'))
	
	img_path = args.demo_path
	Do_show = args.Do_show
	
	# init session
	sess = tf.Session(config=tf.ConfigProto(allow_soft_placement=True))
	# load network
	net = get_network(args.demo_net)
	# load model
	saver = tf.train.Saver(write_version=tf.train.SaverDef.V1)
	saver.restore(sess, args.model)
   
	#sess.run(tf.initialize_all_variables())

	print '\n\nLoaded network {:s}'.format(args.model)

	# Warmup on a dummy image
	im = 128 * np.ones((300, 300, 3), dtype=np.uint8)
	for i in xrange(2):
		_, _= im_detect(sess, net, im)

	# in data/demo/ collect the names of images
	im_names = []
	
	# im_name : 'S10_M0365-02MS.tif'
	for im_name in sorted(os.listdir(img_path)):
		if not im_name.lower().endswith(('.bmp', '.jpeg', '.jpg', '.png', 'tif', '.tiff')):
			continue
		im_names.append(im_name)
		
	if Do_show == 1:
		pass
	else:
		print 'We will not show results.'
	
	CONF_THRESH = [0.0, args.threshSD, args.threshWD]	## vis_detection : thresh is given. So it doesn't work
	
	
	result = {'num_pred_SD':0, 'num_pred_WD':0, 'num_label_SD':0, 'num_label_WD':0,
			  'num_ans_of_pred_SD':0, 'num_ans_of_pred_WD':0, 
			  'num_ans_of_true_SD':0, 'SD_pred_total' : 0}
	# implement and detect defects
	for im_name in im_names:
		print '~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~'
		print 'Demo for data/demo/{}'.format(im_name)
		# im_name : 'S10_M0365-02MS.tif'
		count, pred_roi = split_4_merge(sess, net, im_name, img_path, CONF_THRESH)
		# write result to text file
		f = open(os.path.join(img_path, 'result/'+os.path.splitext(im_name)[0]+'.txt'), 'w')
		for cls in CLASSES[1:]:
			for bdbx in pred_roi[cls]:
				save_info = '{:d} {:d} {:d} {:d} {:s} {:.3f}\r\n'.format(
							int(bdbx[0]), int(bdbx[1]), int(bdbx[2]), int(bdbx[3]), cls, bdbx[4])
				f.write(save_info)
		f.close()
		print('save {:s}'.format(os.path.join(img_path, 'result/'+os.path.splitext(im_name)[0]+'.txt')))
		if len(count) == 0:
			continue	
		#summation of the total results
		for cls in CLASSES[1:]:
			result['num_pred_'+cls] += count['num_pred_'+cls]
			result['num_label_'+cls] += count['num_label_'+cls]
			result['num_ans_of_pred_'+cls] += count['num_ans_of_pred_'+cls]
				
		result['num_ans_of_true_SD'] += count['num_ans_of_true_SD']
		result['SD_pred_total'] += count['SD_pred_total']
	
	#print and write the total results to text_file
	if not len(count) == 0:	   
		pred_SD_true_SD = float(result['num_ans_of_true_SD']) / result['num_label_SD']
		true_over_pred_SD = float(result['num_ans_of_pred_SD']) / result['num_pred_SD']
		
		info1 = 'Sensitivity of SD : {:.3f}'.format(pred_SD_true_SD)   
		info2 = 'Total Sensitivity of SD : {:.3f}'.format(
											float(result['SD_pred_total']) / result['num_label_SD'])
		info3 = 'Precision of SD : {:.3f}'.format(true_over_pred_SD)
		info4 = 'Thresh of SD : {}, Thresh of WD : {}'.format(CONF_THRESH[1], CONF_THRESH[2])
		
		print(info1)
		print(info2)
		print(info3)
		print(info4)
	
		f = open(os.path.join(img_path, 'result/result.txt'), 'a')
		save_info = info1 + '\r\n' + info2 + '\r\n' + info3 + '\r\n' + info4 + '\r\n'
		f.write(save_info)
		for nm in sorted(result.keys()):
			print('The total number of {:s} : {}'.format(nm, result[nm]))
			f.write('The total number of {:s} : {}\r\n'.format(nm, result[nm]))
		f.write('\r\n')
		f.close()
	
	if Do_show == 1:
		plt.show()
	else:
		print 'Do not show results.'