TARGET = many_images_matching
CONFIG += console
CONFIG -= app_bundle
CONFIG += link_pkgconfig
PKGCONFIG += eigen3
TEMPLATE = app

LIBS += -L/usr/lib/
LIBS += `pkg-config \
    opencv \
    --cflags \
    --libs`

SOURCES += main.cpp \
