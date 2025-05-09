import cv2
import numpy as np

class DocumentScanner:
    def __init__(self, output_size=(800, 1100)):
        self.output_size = output_size  # 出力画像のサイズ（幅, 高さ）

    def scan(self, image):
        """画像から四角形を検出してスキャン風に補正する"""
        contour = self._find_document_contour(image)
        if contour is None:
            print("ドキュメントの輪郭が検出できませんでした。")
            return False, image

        warped = self._warp_image_wide(image, contour)
        return True, warped

    def _find_document_contour(self, image):
        """最大の四角形輪郭を探す"""
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        blurred = cv2.GaussianBlur(gray, (5, 5), 0)
        edged = cv2.Canny(blurred, 75, 200)

        contours, _ = cv2.findContours(edged, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
        contours = sorted(contours, key=cv2.contourArea, reverse=True)

        for c in contours:
            peri = cv2.arcLength(c, True)
            approx = cv2.approxPolyDP(c, 0.02 * peri, True)
            if len(approx) == 4:
                # 四角形の輪郭が小さすぎるときはNoneを返す
                if cv2.contourArea(c) < 1000:
                    return None
                return approx.reshape(4, 2)

        return None

    def _warp_image_wide(self, image, pts, margin_ratio=0.05):
        """
        斜め対応：検出された4点をそれぞれ外側に margin_ratio 分だけ押し出して拡張する。
        """
        rect = self._order_points(pts)
        (tl, tr, br, bl, ) = rect
        h, w = image.shape[:2]

        # 各辺のベクトル（単位ベクトル）
        def unit_vector(p1, p2):
            v = p2 - p1
            return v / np.linalg.norm(v)

        # 法線ベクトル（外向き）
        def outward_normal(p1, p2, direction):
            edge = unit_vector(p1, p2)
            if direction == "horizontal":
                return np.array([-edge[1], edge[0]])
            else:
                return np.array([edge[1], -edge[0]])

        # 各辺の長さの5%を margin として使う
        top_len = np.linalg.norm(tr - tl)
        right_len = np.linalg.norm(br - tr)
        bottom_len = np.linalg.norm(br - bl)
        left_len = np.linalg.norm(bl - tl)

        # 各点の移動ベクトルを計算（2辺の法線方向のベクトルの平均）
        def offset_point(p, dir1_vec, dir2_vec, len1, len2):
            margin1 = margin_ratio * len1
            margin2 = margin_ratio * len2
            return p - outward_normal(*dir1_vec, "horizontal") * margin1 \
                    - outward_normal(*dir2_vec, "vertical") * margin2

        # 新しい点の位置（外側に押し出す）
        new_tl = offset_point(tl, (tl, tr), (tl, bl), top_len, left_len)
        new_tr = offset_point(tr, (tr, br), (tr, tl), right_len, top_len)
        new_br = offset_point(br, (br, bl), (br, tr), bottom_len, right_len)
        new_bl = offset_point(bl, (bl, tl), (bl, br), left_len, bottom_len)

        expanded_rect = np.array([new_tl, new_tr, new_br, new_bl], dtype="float32")

        # 変換先の長方形
        dst = np.array([
            [0, 0],
            [self.output_size[0] - 1, 0],
            [self.output_size[0] - 1, self.output_size[1] - 1],
            [0, self.output_size[1] - 1]
        ], dtype="float32")

        # 射影変換行列と変換
        M = cv2.getPerspectiveTransform(expanded_rect, dst)
        warped = cv2.warpPerspective(image, M, self.output_size)

        return warped


    def _warp_image(self, image, pts):
        """4点を元に画像を射影変換して補正"""
        rect = self._order_points(pts)
        (tl, tr, br, bl) = rect

        dst = np.array([
            [0, 0],
            [self.output_size[0] - 1, 0],
            [self.output_size[0] - 1, self.output_size[1] - 1],
            [0, self.output_size[1] - 1]
        ], dtype="float32")

        M = cv2.getPerspectiveTransform(rect, dst)
        warped = cv2.warpPerspective(image, M, self.output_size)

        return warped

    def _order_points(self, pts):
        """左上→右上→右下→左下の順に並べる"""
        rect = np.zeros((4, 2), dtype="float32")
        s = pts.sum(axis=1)
        rect[0] = pts[np.argmin(s)]
        rect[2] = pts[np.argmax(s)]

        diff = np.diff(pts, axis=1)
        rect[1] = pts[np.argmin(diff)]
        rect[3] = pts[np.argmax(diff)]

        return rect
