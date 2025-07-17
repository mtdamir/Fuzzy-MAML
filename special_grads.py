""" Code for second derivatives not implemented in TensorFlow library. """
import tensorflow as tf

@tf.RegisterGradient("MaxPoolGrad")
def _max_pool_grad_grad(op, grad):
    # استخراج ورودی‌ها و ویژگی‌های عملگر
    x = op.inputs[0]  # ورودی اصلی
    output = op.outputs[0]  # خروجی MaxPool
    ksize = op.get_attr("ksize")  # اندازه کرنل
    strides = op.get_attr("strides")  # گام‌ها
    padding = op.get_attr("padding").decode()  # تبدیل به رشته برای سازگاری
    data_format = op.get_attr("data_format").decode() if "data_format" in op.get_attr_map() else "NHWC"

    # محاسبه گرادیان دوم با استفاده از MaxPoolGrad
    gradient = tf.nn.max_pool_grad_v2(x, output, grad, ksize, strides, padding=padding, data_format=data_format)

    # گرادیان‌های صفر برای ورودی‌های دیگر
    gradgrad1 = tf.zeros_like(op.inputs[1], dtype=gradient.dtype)  # گرادیان برای ورودی دوم
    gradgrad2 = tf.zeros_like(op.inputs[2], dtype=gradient.dtype)  # گرادیان برای ورودی سوم

    return (gradient, gradgrad1, gradgrad2)