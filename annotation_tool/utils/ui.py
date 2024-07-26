from PyQt5.QtWidgets import QFileDialog, QLabel, QVBoxLayout, QWidget, QGridLayout, QBoxLayout, \
    QDialogButtonBox, QDialog, QScrollArea, QSizePolicy, QHBoxLayout, QLineEdit
from PyQt5.QtGui import QPixmap, QImage, QPainter, QDoubleValidator
from PyQt5.QtCore import Qt, QSize, pyqtSignal, pyqtSlot, QThread, QPoint
import os
import numpy as np
from utils.io import load_anno
from utils.render import AnnoScene
from time import sleep
from utils.config import Config
    

class Loader(QThread):
    loaded = pyqtSignal(int, QImage)

    def __init__(self, parent=None, **kwargs):
        super().__init__(parent)
        self.running = False
        for k, v in kwargs.items():
            setattr(self, k, v)

    def stop(self):
        self.running = False
        
    def run(self):
        renderer = self.renderer
        for i, img_id in enumerate(self.img_ids):
            if not self.running:
                break
            
            renderer.clear()
            
            qimg = QImage(str(self.rgb_root / ('rgb' + img_id + '.png')))
            
            annotations = load_anno(self.rgb_root.parent / 'pose' / (img_id + '.json'))
            for (start, _), obj in annotations.items():
                renderer.add(obj, start)
            renderer.update()
            
            rendered = renderer.get_rgb(0)[::-1]
            
            painter = QPainter(qimg)
            painter.setRenderHint(QPainter.Antialiasing, False)
            
            painter.setOpacity(0.5)
            painter.drawImage(QPoint(0, 0), QImage(rendered.data.tobytes(), rendered.shape[1], rendered.shape[0], rendered.strides[0], QImage.Format_RGB888))
            painter.end()
            
            self.loaded.emit(i, qimg)
        self.running = False
            

class ImageSelector(QWidget):
    def __init__(self, ncols, rgb_root, img_ids, cam_info, parent=None, img_size=150):
        QWidget.__init__(self, parent=parent)
        self.ncols = ncols
        self.grid_layout = QGridLayout()
        self.grid_layout.setVerticalSpacing(30)
        self.img_size = img_size
        self.renderer = AnnoScene()
        intrinsics = np.array(cam_info['intrinsics'])
        extrinsics = np.array(cam_info['extrinsics'])
        flip_yz = np.eye(4)
        flip_yz[1:3, 1:3] *= -1
        extrinsics = flip_yz @ extrinsics @ flip_yz
        
        # TODO: use a smaller img size
        self.renderer.add_frame(intrinsics, extrinsics, cam_info['width'], cam_info['height'])
        
        self.last_selection = -1
        self.img_ids = img_ids
        self.selections = []

        ## Get all the image files in the directory
        cnt = 0

        ## Render a thumbnail in the widget for every image in the directory
        for img_id in img_ids:
            img_label = QLabel()
            text_label = QLabel()
            img_label.setAlignment(Qt.AlignCenter)
            text_label.setAlignment(Qt.AlignCenter)
            
            pixmap = QPixmap(img_size, img_size * 720 / 1280)
            pixmap.fill()
            img_label.setPixmap(pixmap)
            text_label.setText(img_id)
            img_label.mousePressEvent = \
                lambda e, index=cnt: self.on_thumbnail_click(e, index)
            text_label.mousePressEvent = img_label.mousePressEvent
            thumbnail = QBoxLayout(QBoxLayout.TopToBottom)
            thumbnail.addWidget(img_label)
            thumbnail.addWidget(text_label)
            self.grid_layout.addLayout( \
                thumbnail, cnt // ncols, cnt % ncols, Qt.AlignCenter)

            cnt += 1
            self.selections.append(False)
        self.setLayout(self.grid_layout)
        
        self.loader = Loader(self, rgb_root=rgb_root, img_ids=img_ids, renderer=self.renderer)
        self.loader.loaded.connect(self.on_loaded)
        self.loader.start()
        self.loader.running = True

    def on_thumbnail_click(self, event, index):
        if event.modifiers() & Qt.ShiftModifier and self.last_selection >= 0 and self.last_selection != index:
            all_selected = True
            for i in range(min(self.last_selection, index), max(self.last_selection, index) + 1):
                if not self.selections[i]:
                    all_selected = False
                    break
            for i in range(min(self.last_selection, index), max(self.last_selection, index) + 1):
                self.selections[i] = False if all_selected else True
                text_label_of_thumbnail = self.grid_layout.itemAtPosition(i // self.ncols, i % self.ncols).itemAt(1).widget()
                text_label_of_thumbnail.setStyleSheet("background-color:{};".format('blue' if self.selections[i] else 'none'))
        else:
            self.selections[index] = not self.selections[index]

            text_label_of_thumbnail = self.grid_layout.itemAtPosition(index // self.ncols, index % self.ncols).itemAt(1).widget()
            text_label_of_thumbnail.setStyleSheet("background-color:{};".format('blue' if self.selections[index] else 'none'))

            if self.selections[index]:
                self.last_selection = index
    
    def get_selected_ids(self):
        return [img_id for i, img_id in enumerate(self.img_ids) if self.selections[i]]
    
    def on_loaded(self, idx, qimg):
        img_label = self.grid_layout.itemAtPosition(idx // self.ncols, idx % self.ncols).itemAt(0).widget()
        img_label.setPixmap(QPixmap.fromImage(qimg.scaled(self.img_size, self.img_size, 
                            Qt.AspectRatioMode.KeepAspectRatio, Qt.TransformationMode.SmoothTransformation)))
        img_label.update()
        
    def closeEvent(self, event) -> None:
        self.loader.stop()
        self.loader.quit()
        self.loader.wait()
        

class MessageDialog(QDialog):
    def __init__(self, title, text, parent=None):
        super().__init__(parent)

        self.setWindowTitle(title)

        self.buttonBox = QDialogButtonBox(QDialogButtonBox.Ok)
        self.buttonBox.accepted.connect(self.accept)
        
        self.layout = QVBoxLayout()
        message = QLabel(text)
        self.layout.addWidget(message)
        self.layout.addWidget(self.buttonBox)
        self.setLayout(self.layout)
        

class RunDialog(QDialog):
    def __init__(self, parent=None):
        super().__init__(parent)

        self.setWindowTitle("Parameters")

        QBtn = QDialogButtonBox.Ok | QDialogButtonBox.Cancel

        self.buttonBox = QDialogButtonBox(QBtn)
        self.buttonBox.accepted.connect(self.accept)
        self.buttonBox.rejected.connect(self.reject)

        self.layout = QVBoxLayout()
        def create_option(txt):
            wid = QWidget(self)
            layout = QHBoxLayout()
            layout.addWidget(QLabel(txt))
            edit = QLineEdit()
            edit.setText('0.1')
            edit.setValidator(QDoubleValidator(0, 1, 2))
            layout.addWidget(edit)
            wid.setLayout(layout)
            self.layout.addWidget(wid)
            return edit
        
        self.frac = create_option('Fraction: ')
        
        self.layout.addWidget(self.buttonBox)
        self.setLayout(self.layout)
        
        
def create_image_sel_dialog(self):
    dialog = QDialog(self)
    dialog.setWindowTitle('image selection')

    dialog.buttonBox = QDialogButtonBox(QDialogButtonBox.Ok)
    dialog.buttonBox.accepted.connect(dialog.accept)
    
    img_ids = self.frame_candidates.copy()
    img_ids.remove(self.curr_frame_idx)
    selector = ImageSelector(10, self.scene_root / self.cameras[min(Config.CAMS_TO_ANNO)] / 'rgb_marker', img_ids, self.cam_infos[min(Config.CAMS_TO_ANNO)], dialog)
    selector.setSizePolicy(QSizePolicy.Policy.MinimumExpanding, QSizePolicy.Policy.MinimumExpanding)
    
    scroll = QScrollArea(dialog)
    scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
    scroll.setWidgetResizable(True)
    scroll.setVerticalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAsNeeded)
    scroll.setWidget(selector)
    
    dialog.selector = selector
    
    dialog.layout = QVBoxLayout()
    dialog.layout.addWidget(scroll)
    dialog.layout.addWidget(dialog.buttonBox)
    dialog.setLayout(dialog.layout)
    dialog.resize(selector.ncols * selector.img_size + 100, 600)

    dialog.accepted.connect(lambda: selector.closeEvent(None))
    dialog.closeEvent = lambda _: selector.closeEvent(None)
    
    return dialog
