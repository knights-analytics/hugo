//go:build GO || ALL

package hugot

import (
	"github.com/knights-analytics/hugot/options"
)

func NewGoSession(opts ...options.WithOption) (*Session, error) {
	return newSession("GO", opts...)
}
