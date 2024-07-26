
import json
import numpy as np
import xmltodict
from pathlib import Path
import logging

from utils.obj import ObjectNode, load_obj


def parse_arti_obj_hier(fn):
    anno_file = next(fn.glob('*.urdf'))
    with open(anno_file, 'rb') as fp:
        data_dict = xmltodict.parse(fp.read())['robot']
    links = data_dict['link']
    if isinstance(links, dict):
        logging.warning(f'links should be list type, or there is only one part in object')
    
    nodes = dict()
    for link in links:
        name = link['@name']
        try:
            path = str(link['collision']['geometry']['mesh']['@filename'])[len('package://'):]
        except:
            path = str(link['visual']['geometry']['mesh']['@filename'])[len('package://'):]
        path = fn / path
        node = ObjectNode(None, name)
        nodes[name] = node

    joints = data_dict['joint']
    if isinstance(joints, dict):
        joints = [joints]
        
    for joint in joints:
        parent_name = joint['parent']['@link']
        child_name = joint['child']['@link']
        nodes[parent_name].add_child(nodes[child_name])
    
    for node in nodes.values():
        if node.parent is None:
            return node
    
    
def save_anno(path : Path, anno : dict):
    path.parent.mkdir(parents=True, exist_ok=True)
    res = dict()
    for key, item in anno.items():
        if key[0] == key[1]:
            pose = item.pose
            res[str(key[0])] = dict(name=item.name, path=item.path, articulated=False, m2c_R=pose[:3, :3].tolist(), m2c_t=pose[:3, -1].tolist())
        else:
            poses = dict()
            idx = key[0]
            for node in item:
                pose = node.pose
                poses[idx] = dict(name=node.name, m2c_R=pose[:3, :3].tolist(), m2c_t=pose[:3, -1].tolist())
                idx += 1
            res[str(key[0]) + ',' + str(key[1])] = dict(name=item.name, path=item.path, articulated=True, 
                                                        m2c_R=item.pose[:3, :3].tolist(), m2c_t=item.pose[:3, -1].tolist(), links=poses)

    json.dump(res, open(path, 'w'))


def load_anno(path : Path, read_mesh=True):
    def get_pose(item):
        pose = np.eye(4)
        pose[:3, :3] = np.array(item['m2c_R'])
        pose[:3, -1] = np.array(item['m2c_t'])
        return pose
    
    anno = dict()
    if not path.exists():
        return anno
    content = json.load(open(path, 'r'))
    for key, item in content.items():
        obj = load_obj(Path(item['path'].replace('\\', '/')), read_mesh)
        if item['articulated']:
            links = item['links']
            name2link = dict([(v['name'], v) for v in links.values()])
            def proc_node(node):
                if node.is_leaf():
                    pose = get_pose(name2link[node.name])
                    node.transform(pose @ np.linalg.inv(node.pose))
                    return
                
                if node.name.endswith('__P'):
                    pose = get_pose(name2link[node.name[:-3]])
                    node.transform(pose @ np.linalg.inv(node.pose))
                    
                for child in node.children:
                    proc_node(child)
            obj.transform(get_pose(item))
            proc_node(obj)
        else:
            pose = get_pose(item)
            obj.transform(pose)
        if ',' in key:
            key = tuple([int(num) for num in key.split(',')])
        else:
            key = (int(key), int(key))
        anno[key] = obj
    return anno
    
    

if __name__ == '__main__':
    obj = load_obj(Path('data/models/all/beizi27'))
    anno = dict()
    anno[(1, len(obj))] = obj
    obj2 = load_obj(Path('data/models/all/011_banana.ply'))
    anno[(5,5)] = obj2
    save_anno(Path('test.json'), anno)
    anno = load_anno(Path('test.json'))
    save_anno(Path('test.json'), anno)