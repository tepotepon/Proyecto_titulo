TARGET = many_images_matching
CONFIG += console
CONFIG -= app_bundle
CONFIG += link_pkgconfig
CONFIG += c++11 console
PKGCONFIG += eigen3
TEMPLATE = app

INCLUDEPATH += /usr/local/include/opencv4

LIBS += -L/usr/local/lib -lopencv_core -lopencv_highgui -lopencv_imgproc -lopencv_videoio -lopencv_imgcodecs
LIBS += `pkg-config \
    opencv \
    --cflags \
    --libs`

SOURCES += main.cpp \
