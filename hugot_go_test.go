//go:build GO || ALL

package hugot

import "testing"

// FEATURE EXTRACTION

func TestFeatureExtractionPipelineGo(t *testing.T) {
	session, err := NewGoSession()
	check(t, err)
	defer func(session *Session) {
		destroyErr := session.Destroy()
		check(t, destroyErr)
	}(session)
	featureExtractionPipeline(t, session)
}

func TestFeatureExtractionPipelineValidationGo(t *testing.T) {
	session, err := NewGoSession()
	check(t, err)
	defer func(session *Session) {
		destroyErr := session.Destroy()
		check(t, destroyErr)
	}(session)
	featureExtractionPipelineValidation(t, session)
}

// Text classification

func TestTextClassificationPipelineGo(t *testing.T) {
	session, err := NewGoSession()
	check(t, err)
	defer func(session *Session) {
		destroyErr := session.Destroy()
		check(t, destroyErr)
	}(session)
	textClassificationPipeline(t, session)
}

func TestTextClassificationPipelineMultiGo(t *testing.T) {
	session, err := NewGoSession()
	check(t, err)
	defer func(session *Session) {
		destroyErr := session.Destroy()
		check(t, destroyErr)
	}(session)
	textClassificationPipelineMulti(t, session)
}

func TestTextClassificationPipelineValidationGo(t *testing.T) {
	session, err := NewGoSession()
	check(t, err)
	defer func(session *Session) {
		destroyErr := session.Destroy()
		check(t, destroyErr)
	}(session)
	textClassificationPipelineValidation(t, session)
}

// Token classification

func TestTokenClassificationPipelineGo(t *testing.T) {
	session, err := NewGoSession()
	check(t, err)
	defer func(session *Session) {
		destroyErr := session.Destroy()
		check(t, destroyErr)
	}(session)
	tokenClassificationPipeline(t, session)
}

func TestTokenClassificationPipelineValidationGo(t *testing.T) {
	session, err := NewGoSession()
	check(t, err)
	defer func(session *Session) {
		destroyErr := session.Destroy()
		check(t, destroyErr)
	}(session)
	tokenClassificationPipelineValidation(t, session)
}

// Zero shot

func TestZeroShotClassificationPipelineGo(t *testing.T) {
	session, err := NewGoSession()
	check(t, err)
	defer func(session *Session) {
		destroyErr := session.Destroy()
		check(t, destroyErr)
	}(session)
	zeroShotClassificationPipeline(t, session)
}

func TestZeroShotClassificationPipelineValidationGo(t *testing.T) {
	session, err := NewGoSession()
	check(t, err)
	defer func(session *Session) {
		destroyErr := session.Destroy()
		check(t, destroyErr)
	}(session)
	zeroShotClassificationPipelineValidation(t, session)
}

// No same name

func TestNoSameNamePipelineGo(t *testing.T) {
	session, err := NewGoSession()
	check(t, err)
	defer func(session *Session) {
		destroyErr := session.Destroy()
		check(t, destroyErr)
	}(session)
	noSameNamePipeline(t, session)
}

// Thread safety

func TestThreadSafetyGo(t *testing.T) {
	session, err := NewGoSession()
	check(t, err)
	defer func(session *Session) {
		destroyErr := session.Destroy()
		check(t, destroyErr)
	}(session)
	threadSafety(t, session)
}
