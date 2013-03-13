/* fix for MAC OS where sometime getline() is not supported */

#ifndef __GETLINE_GRAPHCHI_MAXOS_FIX
#define __GETLINE_GRAPHCHI_MAXOS_FIX

/* PASTE AT TOP OF FILE */
#include <stdio.h>   /* flockfile, getc_unlocked, funlockfile */
#include <stdlib.h>  /* malloc, realloc */
#include <errno.h>   /* errno */
#include <unistd.h>  /* ssize_t */

extern "C" ssize_t getline(char **lineptr, size_t *n, FILE *stream);

/* PASTE REMAINDER AT BOTTOM OF FILE */
ssize_t
getline(char **linep, size_t *np, FILE *stream)
{
  char *p = NULL;
  size_t i = 0;

  if (!linep || !np) {
    errno = EINVAL;
    return -1;
  }

  if (!(*linep) || !(*np)) {
    *np = 2400;
    *linep = (char *)malloc(*np);
    if (!(*linep)) {
      return -1;
    }
  }

  flockfile(stream);

  p = *linep;
  for (int ch = 0; (ch = getc_unlocked(stream)) != EOF;) {
    if (i > *np) {
      /* Grow *linep. */
      size_t m = *np * 2;
      char *s = (char *)realloc(*linep, m);

      if (!s) {
        int error = errno;
        funlockfile(stream);
        errno = error;
        return -1;
      }

      *linep = s;
      *np = m;
    }

    p[i] = ch;
    if ('\n' == ch) break;
    i += 1;
  }
  funlockfile(stream);

  /* Null-terminate the string. */
  if (i > *np) {
    /* Grow *linep. */
      size_t m = *np * 2;
      char *s = (char *)realloc(*linep, m);

      if (!s) {
        return -1;
      }

      *linep = s;
      *np = m;
  }

  p[i + 1] = '\0';
  return ((i > 0)? i : -1);
}
#endif //__GETLINE_GRAPHCHI_MAXOS_FIX
