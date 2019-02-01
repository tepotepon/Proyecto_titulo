#-------------------------------------------------
#
# Project created by QtCreator 2017-06-12T10:38:33
#
#-------------------------------------------------

QT       += core gui

greaterThan(QT_MAJOR_VERSION, 4): QT += widgets

TARGET = BINARY-DESCRIPTORS
TEMPLATE = app

LIBS += `pkg-config \
    opencv \
    --cflags \
    --libs`

SOURCES += main.cpp

HEADERS  +=

FORMS    +=
