import cv2
import quality_metrics as qm


im = 'og.png'
ag = qm.avgerage_gradient(im)
ie = qm.information_entropy(im)
uciqe = qm.UCIQE(im)
uiqm = qm.UIQM(im)

print(uiqm)
