from turtle import update
from PyQt5.QtWidgets import QLabel
from PyQt5.QtGui import QImage, QPixmap, QPainter, QColor, QBrush, QPen
from pathlib import Path
import cv2
import PyQt5.QtCore as QtCore
from PyQt5.QtCore import Qt, QObject, QPoint

import numpy as np
import seaborn as sns
from utils.render import compose_img, render_silhouette


class SegScene(QLabel):
    def __init__(self, parent) -> None:
        super().__init__(parent)
        self.seg_palette = sns.color_palette("bright", 64)
        self.setMouseTracking(True)
        self.cand_selection = 0
        self.curr_selection = 0
        self.modifying = 0
        self.control_pressed = False
        self.radius = 10
        self.mouse_pos = None
        
        self.img_shape = None
        
        self.qimg = None
        self.qmask = None
        self.qmask_color = None
        self.qoverlay = None
        self.overlay_buffer = None
    
    def wheelEvent(self, event) -> None:
        self.radius += event.angleDelta().y() // 120
        self.radius = max(2, self.radius)
        self.update()
    
    def getClickedPosition(self, pos):
        contentsRect = QtCore.QRectF(self.contentsRect())
        if pos not in contentsRect:
            return None

        # adjust the position to the contents margins
        pos -= contentsRect.topLeft()
        pos = pos.toPoint() * self.img_scale
        x, y = pos.x(), pos.y()
        if x < self.qmask.width() and y < self.qmask.height() and x >= 0 and y >= 0:
            return pos
        return None
        
    def load_img(self, img):
        self.img_shape = img.shape
        self.qimg = QImage(img.data, img.shape[1], img.shape[0], img.strides[0], QImage.Format_RGB888)
        mask = np.zeros((img.shape[0], img.shape[1]), dtype=np.uint8)
        self.load_mask(mask)
        self.update()
    
    def mousePressEvent(self, event):
        if event.buttons() & QtCore.Qt.LeftButton:
            if event.modifiers() & Qt.ControlModifier and self.curr_selection > 0:
                self.modifying = 1
            else:
                self.curr_selection = self.cand_selection
        elif event.buttons() & QtCore.Qt.RightButton:
            if event.modifiers() & Qt.ControlModifier and self.curr_selection > 0:
                self.modifying = 2
                
    def mouseReleaseEvent(self, event):
        self.modifying = 0
        self.update()
    
    def load_mask_from(self, path : Path):
        if self.qimg is None or not path.exists():
            return
        mask = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
        self.load_mask(mask)
        self.update()
    
    def load_mask(self, mask):
        self.qmask = QImage(mask.data, mask.shape[1], mask.shape[0], mask.strides[0], QImage.Format_Grayscale8)
        self.overlay_buffer = np.zeros((self.qmask.height(), self.qmask.bytesPerLine(), 3), dtype=np.uint8)
        self.qoverlay = QImage(self.overlay_buffer.data, self.overlay_buffer.shape[1], self.overlay_buffer.shape[0], self.overlay_buffer.strides[0], QImage.Format_RGB888)
        
        mask_color = np.zeros((mask.shape[0], mask.shape[1], 3), dtype=np.uint8)
        for i in range(1, mask.max() + 1):
            mask_color[mask == i] = (np.array(self.seg_palette[i - 1]) * 255).astype(np.uint8)
        self.qmask_color = QImage(mask_color.data, mask_color.shape[1], mask_color.shape[0], mask_color.strides[0], QImage.Format_RGB888)
        self.update()
        
    def save_mask(self, path : Path):
        if not path.parent.exists():
            path.parent.mkdir()
        mask_buffer = self.get_mask()
        cv2.imwrite(str(path), mask_buffer)
        
    def get_mask(self):
        ptr = self.qmask.bits()
        ptr.setsize(self.qmask.byteCount())
        mask_buffer = np.frombuffer(ptr, np.uint8).reshape((self.qmask.height(), self.qmask.bytesPerLine()))
        if mask_buffer.shape[1] > self.img_shape[1]:
            mask_buffer = mask_buffer[:, :self.img_shape[1]]
        return mask_buffer
    
    def resizeEvent(self, e) -> None:
        if self.qimg is None:
            return
        self.img_scale = max(self.img_shape[1] / (self.width() - 10), self.img_shape[0] / (self.height() - 10))
        self.update()
        
    def paintEvent(self, e) -> None:
        if self.qimg is None:
            return
        painter = QPainter(self)
        painter.setRenderHint(QPainter.Antialiasing, False)
        painter.drawImage(QPoint(0, 0), self.qimg.scaled(self.width() - 10, self.height() - 10, 
                            Qt.AspectRatioMode.KeepAspectRatio, Qt.TransformationMode.SmoothTransformation))
        
        painter.setOpacity(0.5)
        painter.drawImage(QPoint(0, 0), self.qmask_color.scaled(self.width() - 10, self.height() - 10, 
                                Qt.AspectRatioMode.KeepAspectRatio, Qt.TransformationMode.SmoothTransformation))
        
        if self.mouse_pos is not None:
            painter.setCompositionMode(QPainter.CompositionMode.CompositionMode_Lighten)
            if self.control_pressed and self.curr_selection > 0:
                brush_size = self.radius / self.img_scale
                color = (np.array(self.seg_palette[self.curr_selection - 1]) * 255).astype(np.uint8)
                painter.setBrush(QColor(color[0], color[1], color[2], 255))
                painter.setPen(QPen(QBrush(QColor(color[0], color[1], color[2], 255)), 1.))
                painter.drawEllipse(self.mouse_pos.x() - brush_size / 2, self.mouse_pos.y() - brush_size / 2, brush_size, brush_size)
                
            elif self.cand_selection > 0 or self.curr_selection > 0:
                painter.setOpacity(1.)
                ptr = self.qmask.bits()
                ptr.setsize(self.qmask.byteCount())
                mask_buffer = np.frombuffer(ptr, np.uint8).reshape((self.qmask.height(), self.qmask.bytesPerLine()))
            
                self.overlay_buffer[:] = 0
                if self.curr_selection > 0:
                    sel_mask = render_silhouette(mask_buffer == self.curr_selection, (255, 0, 51), 5)
                    mask = np.any(sel_mask > 0, -1)
                    self.overlay_buffer[mask] = sel_mask[mask]
                
                if self.cand_selection > 0:
                    sel_mask = render_silhouette(mask_buffer == self.cand_selection, (255, 204, 204), 5)
                    mask = np.any(sel_mask > 0, -1)
                    self.overlay_buffer[mask] = sel_mask[mask]
                painter.drawImage(QPoint(0, 0), self.qoverlay.scaled(self.width() - 10, self.height() - 10, 
                                Qt.AspectRatioMode.KeepAspectRatio, Qt.TransformationMode.FastTransformation))

        painter.end()
        
    def mouseMoveEvent(self, e):
        if self.qimg is None:
            return
        
        self.mouse_pos = e.pos()
        pos = self.getClickedPosition(e.pos())
        if pos is not None:
            if self.modifying > 0:
                selid = self.curr_selection if self.modifying == 1 else 0
                
                painter = QPainter(self.qmask)
                painter.setRenderHint(QPainter.Antialiasing, False)
                brush = QBrush(QColor(selid, selid, selid, 255))
                painter.setPen(QPen(brush, 1.))
                painter.setBrush(brush)
                painter.drawEllipse(pos.x() - self.radius / 2, pos.y() - self.radius / 2, self.radius, self.radius)
                painter.end()
                
                painter = QPainter(self.qmask_color)
                painter.setRenderHint(QPainter.Antialiasing, False)
                color = (np.array(self.seg_palette[selid - 1]) * 255).astype(np.uint8) if selid > 0 else (0, 0, 0)
                brush = QBrush(QColor(color[0], color[1], color[2], 255))
                painter.setPen(QPen(brush, 1.))
                painter.setBrush(brush)
                painter.drawEllipse(pos.x() - self.radius / 2, pos.y() - self.radius / 2, self.radius, self.radius)
                painter.end()
            else:
                self.cand_selection = QColor(self.qmask.pixel(pos.x(), pos.y())).red()
            self.update()
            
    def remove_anno(self, key):
        ptr = self.qmask.bits()
        ptr.setsize(self.qmask.byteCount())
        mask = np.frombuffer(ptr, np.uint8).reshape((self.qmask.height(), self.qmask.bytesPerLine()))
        for i in range(key[0], key[1] + 1):
            mask[mask == i] = 0
        
        mask_color = np.zeros((mask.shape[0], mask.shape[1], 3), dtype=np.uint8)
        for i in range(1, mask.max() + 1):
            mask_color[mask == i] = (np.array(self.seg_palette[i - 1]) * 255).astype(np.uint8)
        self.qmask_color = QImage(mask_color.data, mask_color.shape[1], mask_color.shape[0], mask_color.strides[0], QImage.Format_RGB888)
        self.update()
        
    def keyPressEvent(self, e) -> None:
        key = e.key()
        if key == Qt.Key.Key_Control:
            self.control_pressed = True
            self.update()
    
    def keyReleasedEvent(self, e) -> None:
        key = e.key()
        if key == Qt.Key.Key_Control:
            self.control_pressed = False
            self.update()
    