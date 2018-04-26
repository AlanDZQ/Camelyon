from utils import TRAIN_TUMOR_WSI_PATH, TRAIN_NORMAL_WSI_PATH, TRAIN_TUMOR_MASK_PATH
from utils import PROCESSED_PATCHES_NORMAL_NEGATIVE_PATH, PROCESSED_PATCHES_TUMOR_NEGATIVE_PATH
from utils import PROCESSED_PATCHES_POSITIVE_PATH,PROCESSED_PATCHES_FROM_USE_MASK_POSITIVE_PATH
from utils import PATCH_SIZE, PATCH_NORMAL_PREFIX, PATCH_TUMOR_PREFIX
from openslide import OpenSlide, OpenSlideUnsupportedFormatError
from PIL import Image
import glob
import numpy as np
import cv2
import os



class WSI(object):
    """
        # ================================
        # 这个类用于提取 Patch
        # ================================
    """
    index = 0
    negative_patch_index = 118456
    positive_patch_index = 2230
    wsi_paths = []
    mask_paths = []
    def_level = 7
    cur_wsi_path = ''
    wsi_image = ''
    mask_image = ''
    level_used = ''
    mask_pil = ''
    mask = ''


    def read_wsi_mask(self, wsi_path, mask_path):
        """

        :param wsi_path: 传入 .tif 文件路径 该文件是tumor图片
        :param mask_path: 传入 .tif 文件路径 该文件是mask图片 是黑白图像，白色部分代表肿瘤区域
        :return:
        """
        try:
            self.cur_wsi_path = wsi_path
            self.wsi_image = OpenSlide(wsi_path)  # 读取 .tif  作为一个文件对象
            self.mask_image = OpenSlide(mask_path)

            """
                    使用最小的分辨率 
                    0 为最高分辨率 level_count - 1 为最小分辨率 在这里 level_use 为默认7(1024,700)
            """
            self.level_used = min(self.def_level, self.wsi_image.level_count - 1, self.mask_image.level_count - 1)
            """
                read_region(location, level, size)
                Return an RGBA Image containing the contents of the specified region.
                返回的是一个 rgba四通道图像
                这个代表是读取 level7 的一整幅图像 返回的是一个PIL对象(height=700,width=1024)的四通道对象
            """
            self.mask_pil = self.mask_image.read_region((0, 0), self.level_used,
                                                        self.mask_image.level_dimensions[self.level_used])

            self.mask = np.array(self.mask_pil)

        except OpenSlideUnsupportedFormatError:
            print('Exception: OpenSlideUnsupportedFormatError')
            return False

        return True

    def read_wsi_normal(self, wsi_path):
        """
            # =====================================================================================
            # 通过openslide 读取图像到 类中
            # Due to memory constraint, we use down sampled (4th level, 1/32 resolution) image
            # ======================================================================================
        """
        try:
            self.cur_wsi_path = wsi_path
            self.wsi_image = OpenSlide(wsi_path)
            self.level_used = min(self.def_level, self.wsi_image.level_count - 1)

            self.rgb_image_pil = self.wsi_image.read_region((0, 0), self.level_used,
                                                            self.wsi_image.level_dimensions[self.level_used])
            self.rgb_image = np.array(self.rgb_image_pil)

        except OpenSlideUnsupportedFormatError:
            print('Exception: OpenSlideUnsupportedFormatError')
            return False

        return True

    def read_wsi_tumor(self, wsi_path, mask_path):
        """
            # =====================================================================================
            # read WSI image and resize
            # Due to memory constraint, we use down sampled (4th level, 1/32 resolution) image
            # ======================================================================================
        """
        try:
            self.cur_wsi_path = wsi_path
            self.wsi_image = OpenSlide(wsi_path)
            self.mask_image = OpenSlide(mask_path)

            self.level_used = min(self.def_level, self.wsi_image.level_count - 1, self.mask_image.level_count - 1)

            self.rgb_image_pil = self.wsi_image.read_region((0, 0), self.level_used,
                                                            self.wsi_image.level_dimensions[self.level_used])
            self.rgb_image = np.array(self.rgb_image_pil)

        except OpenSlideUnsupportedFormatError:
            print('Exception: OpenSlideUnsupportedFormatError')
            return False

        return True

    def find_roi_n_extract_patches_mask(self):
        """
            使用Opencv转化为二值化灰度图 使用 findcontours 找轮廓 最后 找到外接矩形框 并进一步处理
        """
        """
            self.mask shape(700,1024,4)
            颜色空间转换之后 mask.shape = (700,1024)
        """
        mask = cv2.cvtColor(self.mask, cv2.COLOR_BGR2GRAY) # 颜色空间转变
        """
            参数 cv2.RETR_EXTERNAL表示只检测外轮廓
            
        """
        _, contours, _ = cv2.findContours(np.array(mask), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE) # contours为点集对象
        bounding_boxes = [cv2.boundingRect(c) for c in contours] # bounding_boxes 为外接矩形集合

        self.mask_pil.close()
        self.extract_patches_mask(bounding_boxes)
        self.wsi_image.close()
        self.mask_image.close()

    def find_roi_n_extract_patches_normal(self):
        hsv = cv2.cvtColor(self.rgb_image, cv2.COLOR_BGR2HSV)
        # [20, 20, 20]
        lower_red = np.array([20, 50, 20])
        # [255, 255, 255]
        upper_red = np.array([200, 150, 200])
        mask = cv2.inRange(hsv, lower_red, upper_red)

        # (50, 50)
        close_kernel = np.ones((25, 25), dtype=np.uint8) # 闭运算的核
        image_close = Image.fromarray(cv2.morphologyEx(np.array(mask), cv2.MORPH_CLOSE, close_kernel))
        # (30, 30)
        open_kernel = np.ones((30, 30), dtype=np.uint8)
        image_open = Image.fromarray(cv2.morphologyEx(np.array(image_close), cv2.MORPH_OPEN, open_kernel))

        _, contours, _ = cv2.findContours(np.array(image_open), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        bounding_boxes = [cv2.boundingRect(c) for c in contours]

        self.rgb_image_pil.close()
        self.extract_patches_normal(bounding_boxes)
        self.wsi_image.close()

    def find_roi_n_extract_patches_tumor(self):
        hsv = cv2.cvtColor(self.rgb_image, cv2.COLOR_BGR2HSV)
        lower_red = np.array([20, 20, 20])
        upper_red = np.array([255, 255, 255])
        mask = cv2.inRange(hsv, lower_red, upper_red)

        # (50, 50)
        close_kernel = np.ones((50, 50), dtype=np.uint8)
        image_close = Image.fromarray(cv2.morphologyEx(np.array(mask), cv2.MORPH_CLOSE, close_kernel))
        # (30, 30)
        open_kernel = np.ones((30, 30), dtype=np.uint8)
        image_open = Image.fromarray(cv2.morphologyEx(np.array(image_close), cv2.MORPH_OPEN, open_kernel))
        _, contours, _ = cv2.findContours(np.array(image_open), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        bounding_boxes = [cv2.boundingRect(c) for c in contours]
        # contour_rgb = cv2.resize(contour_rgb, (0, 0), fx=0.40, fy=0.40)
        # cv2.imshow('contour_rgb', np.array(contour_rgb))
        self.rgb_image_pil.close()
        self.extract_patches_tumor(bounding_boxes)
        self.wsi_image.close()
        self.mask_image.close()

    def extract_patches_mask(self, bounding_boxes):
        """
        提取 label 为1 的样本

        :param bounding_boxes: 通过 contours 的外接矩形列表
        :return:

        """
        mag_factor = pow(2, self.level_used)  # 缩放到 0 level的比例

        print('No. of ROIs to extract patches from: %d' % len(bounding_boxes))

        for i, bounding_box in enumerate(bounding_boxes):
            b_x_start = int(bounding_box[0]) * mag_factor  # 对应到level0 上面的 x
            b_y_start = int(bounding_box[1]) * mag_factor  # 同理
            b_x_end = (int(bounding_box[0]) + int(bounding_box[2])) * mag_factor
            b_y_end = (int(bounding_box[1]) + int(bounding_box[3])) * mag_factor
            X = np.random.random_integers(b_x_start, high=b_x_end, size=500)
            Y = np.random.random_integers(b_y_start, high=b_y_end, size=500)

            for x, y in zip(X, Y):
                mask = self.mask_image.read_region((x, y), 0, (PATCH_SIZE, PATCH_SIZE)) #在level 0 上面去截取图片
                mask_gt = np.array(mask)
                mask_gt = cv2.cvtColor(mask_gt, cv2.COLOR_BGR2GRAY)

                white_pixel_cnt_gt = cv2.countNonZero(mask_gt)  # mask中的非0的像素个数 (肿瘤像素所占的个数)

                if white_pixel_cnt_gt > ((PATCH_SIZE * PATCH_SIZE) * 0.90):  # (如果白色像素个数在掩码中所占的比例超过90%)
                    # mask = Image.fromarray(mask)
                    # 读取这个图片并且保存为png格式
                    patch = self.wsi_image.read_region((x, y), 0, (PATCH_SIZE, PATCH_SIZE))
                    patch.save(PROCESSED_PATCHES_FROM_USE_MASK_POSITIVE_PATH + PATCH_TUMOR_PREFIX +
                               str(self.positive_patch_index), 'PNG')
                    self.positive_patch_index += 1
                    patch.close()

                mask.close()

    def extract_patches_normal(self, bounding_boxes):
        """
            从normal中提取label-0的训练样本 在boundingboxes 中随机提取

            :param bounding_boxes: list of bounding boxes corresponds to detected ROIs
            :return:

        """
        mag_factor = pow(2, self.level_used)

        print('No. of ROIs to extract patches from: %d' % len(bounding_boxes))

        for i, bounding_box in enumerate(bounding_boxes):
            b_x_start = int(bounding_box[0]) * mag_factor
            b_y_start = int(bounding_box[1]) * mag_factor
            b_x_end = (int(bounding_box[0]) + int(bounding_box[2])) * mag_factor
            b_y_end = (int(bounding_box[1]) + int(bounding_box[3])) * mag_factor
            X = np.random.random_integers(b_x_start, high=b_x_end, size=500)
            Y = np.random.random_integers(b_y_start, high=b_y_end, size=500)

            for x, y in zip(X, Y):
                patch = self.wsi_image.read_region((x, y), 0, (PATCH_SIZE, PATCH_SIZE))
                patch_array = np.array(patch)

                patch_hsv = cv2.cvtColor(patch_array, cv2.COLOR_BGR2HSV)
                # [20, 20, 20]
                lower_red = np.array([20, 20, 20])
                # [255, 255, 255]
                upper_red = np.array([200, 200, 200])
                mask = cv2.inRange(patch_hsv, lower_red, upper_red)
                white_pixel_cnt = cv2.countNonZero(mask)

                if white_pixel_cnt > ((PATCH_SIZE * PATCH_SIZE) * 0.50):
                    # mask = Image.fromarray(mask)
                    patch.save(PROCESSED_PATCHES_NORMAL_NEGATIVE_PATH + PATCH_NORMAL_PREFIX +
                               str(self.negative_patch_index), 'PNG')
                    # mask.save(PROCESSED_PATCHES_NORMAL_PATH + PATCH_NORMAL_PREFIX + str(self.patch_index),
                    #           'PNG')
                    self.negative_patch_index += 1

                patch.close()

    def extract_patches_tumor(self, bounding_boxes):
        """
            提取带有肿瘤切片的label-1 和 label-0样本
            :param bounding_boxes: list of bounding boxes corresponds to detected ROIs
            :return:
            
        """
        mag_factor = pow(2, self.level_used)  # 缩放到 0 level的比例

        print('No. of ROIs to extract patches from: %d' % len(bounding_boxes))

        for i, bounding_box in enumerate(bounding_boxes):
            b_x_start = int(bounding_box[0]) * mag_factor
            b_y_start = int(bounding_box[1]) * mag_factor
            b_x_end = (int(bounding_box[0]) + int(bounding_box[2])) * mag_factor
            b_y_end = (int(bounding_box[1]) + int(bounding_box[3])) * mag_factor
            X = np.random.random_integers(b_x_start, high=b_x_end, size=500)
            Y = np.random.random_integers(b_y_start, high=b_y_end, size=500)
            # X = np.arange(b_x_start, b_x_end-256, 5)
            # Y = np.arange(b_y_start, b_y_end-256, 5)

            for x, y in zip(X, Y):
                patch = self.wsi_image.read_region((x, y), 0, (PATCH_SIZE, PATCH_SIZE))
                mask = self.mask_image.read_region((x, y), 0, (PATCH_SIZE, PATCH_SIZE))
                mask_gt = np.array(mask)
                # mask_gt = cv2.cvtColor(mask_gt, cv2.COLOR_BGR2GRAY)
                mask_gt = cv2.cvtColor(mask_gt, cv2.COLOR_BGR2GRAY)
                patch_array = np.array(patch)

                white_pixel_cnt_gt = cv2.countNonZero(mask_gt)
                """
                    如果这个肿瘤图片的这个区域中没有肿瘤部分
                """
                if white_pixel_cnt_gt == 0:  # mask_gt does not contain tumor area
                    patch_hsv = cv2.cvtColor(patch_array, cv2.COLOR_BGR2HSV)
                    lower_red = np.array([20, 20, 20])
                    upper_red = np.array([200, 200, 200])
                    mask_patch = cv2.inRange(patch_hsv, lower_red, upper_red)
                    white_pixel_cnt = cv2.countNonZero(mask_patch)
                    # 选取颜色空间在这个范围的部分如果白色像素所占的比例超过0.5那么我们把它保存
                    if white_pixel_cnt > ((PATCH_SIZE * PATCH_SIZE) * 0.50):
                        # mask = Image.fromarray(mask)
                        patch.save(PROCESSED_PATCHES_TUMOR_NEGATIVE_PATH + PATCH_NORMAL_PREFIX +
                                   str(self.negative_patch_index), 'PNG')
                        # mask.save(PROCESSED_PATCHES_NORMAL_PATH + PATCH_NORMAL_PREFIX + str(self.patch_index),
                        #           'PNG')
                        self.negative_patch_index += 1
                else:  # mask_gt contains tumor area #如果这个肿瘤区域中有肿瘤的部分大于85%
                    if white_pixel_cnt_gt >= ((PATCH_SIZE * PATCH_SIZE) * 0.85):
                        patch.save(PROCESSED_PATCHES_POSITIVE_PATH + PATCH_TUMOR_PREFIX +
                                   str(self.positive_patch_index), 'PNG')
                        self.positive_patch_index += 1

                patch.close()
                mask.close()


def run_on_mask_data():
    wsi.wsi_paths = glob.glob(os.path.join(TRAIN_TUMOR_WSI_PATH, '*.tif'))
    wsi.wsi_paths.sort()
    wsi.mask_paths = glob.glob(os.path.join(TRAIN_TUMOR_MASK_PATH, '*.tif'))
    wsi.mask_paths.sort()

    wsi.index = 0

    for wsi_path, mask_path in zip(wsi.wsi_paths, wsi.mask_paths):
        if wsi.read_wsi_mask(wsi_path, mask_path):  # 读入mask图片和对应的tumor图片
            wsi.find_roi_n_extract_patches_mask()  # 提取roi 并且保存


def run_on_tumor_data():
    wsi.wsi_paths = glob.glob(os.path.join(TRAIN_TUMOR_WSI_PATH, '*.tif'))
    wsi.wsi_paths.sort()
    wsi.mask_paths = glob.glob(os.path.join(TRAIN_TUMOR_MASK_PATH, '*.tif'))
    wsi.mask_paths.sort()

    wsi.index = 0

    for wsi_path, mask_path in zip(wsi.wsi_paths, wsi.mask_paths):
        if wsi.read_wsi_tumor(wsi_path, mask_path):  # 读入tumor图片和mask图片 到 wsi类中
            wsi.find_roi_n_extract_patches_tumor()  #


def run_on_normal_data():
    wsi.wsi_paths = glob.glob(os.path.join(TRAIN_NORMAL_WSI_PATH, '*.tif'))  # 加载normal文件的路径
    wsi.wsi_paths.sort()

    wsi.index = 0

    for wsi_path in wsi.wsi_paths:
        if wsi.read_wsi_normal(wsi_path):  # 首先把wsi文件读入到类中
            wsi.find_roi_n_extract_patches_normal()  # 在这里提取patch


if __name__ == '__main__':
    wsi = WSI()
    #run_on_tumor_data()  # 读入mask和tumor tumor 图片提取 非肿瘤部分
    #run_on_normal_data()  # 对 normal 图片提取 正常部分
    run_on_mask_data()  # 对 读入mask和tumor图片 提取肿瘤部分
