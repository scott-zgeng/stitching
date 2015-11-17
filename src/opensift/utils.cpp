/*
Miscellaneous utility functions.

Copyright (C) 2006-2012  Rob Hess <rob@iqengines.com>

@version 1.1.2-20100521
*/

#include "utils.h"

//#include <cv.h>
//#include <cxcore.h>
//#include <highgui.h>

//#include <gdk/gdk.h>
//#include <gtk/gtk.h>

#include <errno.h>
#include <string.h>
#include <stdlib.h>
#include <stdarg.h>
#include <time.h>

#include "../mv_base.h"




#define MAX_LOG_LEN 512

/*************************** Function Definitions ****************************/

static inline char* time2str(char* ptr)
{
    time_t t = time(NULL);
    tm* curr_time;
    curr_time = localtime(&t);
    sprintf(ptr, "%02d:%02d:%02d", curr_time->tm_hour, curr_time->tm_min, curr_time->tm_sec);
    return ptr + 8;
}

static inline char* level2str(char* ptr, int level)
{
    static char log_info[][4] = { "NON", "INF", "WRN", "ERR" };
    memcpy(ptr, log_info[level], 3);
    return ptr + 3;
}

static inline char* file2str(char* ptr, const char* file)
{
    const char* end = file;
    while (*end != '\0')  end++;

    const char* start = end;
    while (start != file && *start != '\\' && *start != '/')  start--;
    if (*start == '\\' || *start == '/') start++;

    int len = end - start;
    if (len <= 12) {        
        memcpy(ptr, start, len);        
    } else {
        *ptr++ = '~';
        memcpy(ptr, end - 11, 11);
        len = 12;
    }

    return ptr + len;
}

static inline char* line2str(char* ptr, int line)
{
    int n = line % 100000;
    *ptr++ = '0' + (n / 10000);
    n = n % 10000;
    *ptr++ = '0' + (n / 1000);
    n = n % 1000;
    *ptr++ = '0' + (n / 100);
    n = n % 100;
    *ptr++ = '0' + (n / 10);
    n = n % 10;
    *ptr++ = '0' + (n);
    return ptr;
}


void write_log(int level, const char* file, int line, const char* format, ...)
{
    char log[MAX_LOG_LEN + 32];
    char* ptr = log;

    ptr = time2str(ptr);
    *ptr++ = '|';
    ptr = level2str(ptr, level);   
    *ptr++ = '|';
    ptr = file2str(ptr, file);       
    *ptr++ = ':';
    ptr = line2str(ptr, line);
    *ptr++ = '|';

    va_list ap;
    va_start(ap, format);
    vsnprintf(ptr, MAX_LOG_LEN, format, ap);
    va_end(ap);

    printf(log);
    printf("\n");
}


/*
  Replaces a file's extension, which is assumed to be everything after the
  last dot ('.') character.

  @param file the name of a file

  @param extn a new extension for \a file; should not include a dot (i.e.
  \c "jpg", not \c ".jpg") unless the new file extension should contain
  two dots.

  @return Returns a new string formed as described above.  If \a file does
  not have an extension, this function simply adds one.
  */
char* replace_extension(const char* file, const char* extn)
{
    char* new_file, *lastdot;

    new_file = (char*)calloc(strlen(file) + strlen(extn) + 2, sizeof(char));
    strcpy(new_file, file);
    lastdot = strrchr(new_file, '.');
    if (lastdot)
        *(lastdot + 1) = '\0';
    else
        strcat(new_file, ".");
    strcat(new_file, extn);

    return new_file;
}



/*
  Prepends a path to a filename.

  @param path a path
  @param file a file name

  @return Returns a new string containing a full path name consisting of
  \a path prepended to \a file.
  */
//char* prepend_path(const char* path, const char* file)
//{
//    int n = strlen(path) + strlen(file) + 2;
//    char* pathname = (char*)calloc(n, sizeof(char));
//
//    snprintf(pathname, n, "%s/%s", path, file);
//
//    return pathname;
//}



/*
  A function that removes the path from a filename.  Similar to the Unix
  basename command.

  @param pathname a (full) path name

  @return Returns the basename of \a pathname.
  */
//char* basename(const char* pathname)
//{
//    char* base, *last_slash;
//
//    last_slash = strrchr(pathname, '/');
//    if (!last_slash)
//    {
//        base = calloc(strlen(pathname) + 1, sizeof(char));
//        strcpy(base, pathname);
//    }
//    else
//    {
//        base = calloc(strlen(last_slash++), sizeof(char));
//        strcpy(base, last_slash);
//    }
//
//    return base;
//}



/*
  Displays progress in the console with a spinning pinwheel.  Every time this
  function is called, the state of the pinwheel is incremented.  The pinwheel
  has four states that loop indefinitely: '|', '/', '-', '\'.

  @param done if 0, this function simply increments the state of the pinwheel;
  otherwise it prints "done"
  */
//void progress(int done)
//{
//    char state[4] = { '|', '/', '-', '\\' };
//    static int cur = -1;
//
//    if (cur == -1)
//        fprintf(stderr, "  ");
//
//    if (done)
//    {
//        fprintf(stderr, "\b\bdone\n");
//        cur = -1;
//    }
//    else
//    {
//        cur = (cur + 1) % 4;
//        fprintf(stdout, "\b\b%c ", state[cur]);
//        fflush(stderr);
//    }
//}



/*
  Erases a specified number of characters from a stream.

  @param stream the stream from which to erase characters
  @param n the number of characters to erase
  */
//void erase_from_stream(FILE* stream, int n)
//{
//    int j;
//    for (j = 0; j < n; j++)
//        fprintf(stream, "\b");
//    for (j = 0; j < n; j++)
//        fprintf(stream, " ");
//    for (j = 0; j < n; j++)
//        fprintf(stream, "\b");
//}
//


/*
  Doubles the size of an array with error checking

  @param array pointer to an array whose size is to be doubled
  @param n number of elements allocated for \a array
  @param size size in bytes of elements in \a array

  @return Returns the new number of elements allocated for \a array.  If no
  memory is available, returns 0.
  */
int array_double(void** array, int n, int size)
{
    void* tmp;

    tmp = realloc(*array, 2 * n * size);
    if (!tmp)
    {
        fprintf(stderr, "Warning: unable to allocate memory in array_double(),"
            " %s line %d\n", __FILE__, __LINE__);
        if (*array)
            free(*array);
        *array = NULL;
        return 0;
    }
    *array = tmp;
    return n * 2;
}



/*
  Calculates the squared distance between two points.

  @param p1 a point
  @param p2 another point
  */
double dist_sq_2D(mv_point_d_t p1, mv_point_d_t p2)
{
    double x_diff = p1.x - p2.x;
    double y_diff = p1.y - p2.y;

    return x_diff * x_diff + y_diff * y_diff;
}



/*
  Draws an x on an image.

  @param img an image
  @param pt the center point of the x
  @param r the x's radius
  @param w the x's line weight
  @param color the color of the x
  */
//void draw_x(mv_image_t* img, mv_point_t pt, int r, int w, mv_scalar_t color)
//{
//    mv_line(img, pt, mv_point_t(pt.x + r, pt.y + r), color, w, 8, 0);
//    mv_line(img, pt, mv_point_t(pt.x - r, pt.y + r), color, w, 8, 0);
//    mv_line(img, pt, mv_point_t(pt.x + r, pt.y - r), color, w, 8, 0);
//    mv_line(img, pt, mv_point_t(pt.x - r, pt.y - r), color, w, 8, 0);
//}



/*
  Combines two images by scacking one on top of the other

  @param img1 top image
  @param img2 bottom image

  @return Returns the image resulting from stacking \a img1 on top if \a img2
  */
//extern mv_image_t* stack_imgs(mv_image_t* img1, mv_image_t* img2)
//{
//    mv_image_t* stacked = mv_create_image(mv_size_t(MAX(img1->width, img2->width),
//        img1->height + img2->height),
//        IPL_DEPTH_8U, 3);
//
//    mv_set_zero(stacked);
//    mv_set_image_roi(stacked, mv_rect_t(0, 0, img1->width, img1->height));
//    mv_add(img1, stacked, stacked, NULL);
//    mv_set_image_roi(stacked, mv_rect_t(0, img1->height, img2->width, img2->height));
//    mv_add(img2, stacked, stacked, NULL);
//    mv_reset_image_roi(stacked);
//
//    return stacked;
//}


//extern mv_image_t* stack_imgs_horizontal(mv_image_t* img1, mv_image_t* img2)
//{
//    
//    mv_image_t * stacked = mv_create_image(mv_size_t(img1->width + img2->width, MAX(img1->height, img2->height)),
//        IPL_DEPTH_8U, 3);
//    mv_set_zero(stacked);
//    mv_set_image_roi(stacked, mv_rect_t(0, 0, img1->width, img1->height));
//    mv_add(img1, stacked, stacked, NULL);
//    mv_set_image_roi(stacked, mv_rect_t(img1->width, 0, img2->width, img2->height));
//    mv_add(img2, stacked, stacked, NULL);
//    mv_reset_image_roi(stacked);
//
//    return stacked;
//}


/*
  Displays an image, making sure it fits on screen.  cvWaitKey() must be
  called after this function so the event loop is entered and the
  image is displayed.

  @param img an image, possibly too large to display on-screen
  @param title the title of the window in which \a img is displayed
  */
//void display_big_img(mv_image_t* img, char* title)
//{
//    mv_image_t* small;
//    //GdkScreen* scr;
//    int scr_width, scr_height;
//    double img_aspect, scr_aspect, scale;
//
//    /* determine screen size to see if image fits on screen */
//    //gdk_init(NULL, NULL);
//    //scr = gdk_screen_get_default();
//    //scr_width = gdk_screen_get_width(scr);
//    //scr_height = gdk_screen_get_height(scr);
//    scr_width = 1680;
//    scr_height = 1050;
//
//
//    if (img->width >= 0.90 * scr_width || img->height >= 0.90 * scr_height)
//    {
//        img_aspect = (double)(img->width) / img->height;
//        scr_aspect = (double)(scr_width) / scr_height;
//
//        if (img_aspect > scr_aspect)
//            scale = 0.90 * scr_width / img->width;
//        else
//            scale = 0.90 * scr_height / img->height;
//
//        small = mv_create_image(mv_size_t(img->width * scale, img->height * scale),
//            img->depth, img->nChannels);
//        mv_resize(img, small, MV_INTER_AREA);
//    }
//    else
//        small = mv_clone_image(img);
//
//    mv_named_window(title);
//    mv_show_image(title, small);
//    mv_release_image(&small);
//}



/*
  Allows user to view an array of images as a video.  Keyboard controls
  are as follows:

  <ul>
  <li>Space - start and pause playback</li>
  <li>Page Down - skip forward 10 frames</li>
  <li>Page Up - jump back 10 frames</li>
  <li>Right Arrow - skip forward 1 frame</li>
  <li>Left Arrow - jump back 1 frame</li>
  <li>Backspace - jump back to beginning</li>
  <li>Esc - exit playback</li>
  <li>Closing the window also exits playback</li>
  </ul>

  @param imgs an array of images
  @param n number of images in \a imgs
  @param win_name name of window in which images are displayed
//  */
//void vid_view(mv_image_t** imgs, int n, char* win_name)
//{
//    int k, i = 0, playing = 0;
//
//    display_big_img(imgs[i], win_name);
//    while (!win_closed(win_name))
//    {
//        /* if already playing, advance frame and check for pause */
//        if (playing)
//        {
//            i = MIN(i + 1, n - 1);
//            display_big_img(imgs[i], win_name);
//            k = mv_wait_key(33);
//            if (k == ' ' || i == n - 1)
//                playing = 0;
//        }
//
//        else
//        {
//            k = mv_wait_key(0);
//            
//            switch (k)
//            {
//                /* space */
//            case ' ':
//                playing = 1;
//                break;
//
//                /* esc */
//            case 27:
//            case 1048603:
//                mv_destroy_window(win_name);
//                break;
//
//                /* backspace */
//            case '\b':
//                i = 0;
//                display_big_img(imgs[i], win_name);
//                break;
//
//                /* left arrow */
//            case 65288:
//            case 1113937:
//                i = MAX(i - 1, 0);
//                display_big_img(imgs[i], win_name);
//                break;
//
//                /* right arrow */
//            case 65363:
//            case 1113939:
//                i = MIN(i + 1, n - 1);
//                display_big_img(imgs[i], win_name);
//                break;
//
//                /* page up */
//            case 65365:
//            case 1113941:
//                i = MAX(i - 10, 0);
//                display_big_img(imgs[i], win_name);
//                break;
//
//                /* page down */
//            case 65366:
//            case 1113942:
//                i = MIN(i + 10, n - 1);
//                display_big_img(imgs[i], win_name);
//                break;
//            }
//        }
//    }
//}
//


/*
  Checks if a HighGUI window is still open or not

  @param name the name of the window we're checking

  @return Returns 1 if the window named \a name has been closed or 0 otherwise
  */
//int win_closed(char* win_name)
//{
//    if (!mv_get_windows_handle(win_name))
//        return 1;
//    return 0;
//}
