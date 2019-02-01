QT += core
QT -= gui

TARGET = input_output
CONFIG += console
CONFIG -= app_bundle

TEMPLATE = app

SOURCES += main.cpp \
    view.cpp

LIBS += `pkg-config \
    opencv \
    --cflags \
    --libs`

HEADERS += \
    view.h


