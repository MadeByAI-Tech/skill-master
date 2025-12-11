from pipeline.processing_job import extract

def test_processing_job_extract():
    assert extract("abcd") == "abcd-extracted"