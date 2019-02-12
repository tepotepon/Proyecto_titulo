QT += core
QT -= gui

TARGET = many_images_matching
CONFIG += console
CONFIG -= app_bundle

TEMPLATE = app

LIBS += `pkg-config \
    opencv \
    --cflags \
    --libs`

SOURCES += main.cpp \
    view.cpp

HEADERS += \
    stats.h \
    utils.h \
    many_images_matching.h \
    view.h \
    sfm.h \
    cloudpoint.h

