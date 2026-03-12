"""Scrapy settings for the regulatory harvesting project."""

BOT_NAME = "legal_agent"
SPIDER_MODULES = ["legal_agent.scraping.spiders"]
NEWSPIDER_MODULE = "legal_agent.scraping.spiders"

ROBOTSTXT_OBEY = True
CONCURRENT_REQUESTS = 4
DOWNLOAD_DELAY = 2
COOKIES_ENABLED = False

DEFAULT_REQUEST_HEADERS = {
    "Accept": "text/html,application/xhtml+xml,application/pdf",
    "Accept-Language": "en",
}

ITEM_PIPELINES = {
    "legal_agent.scraping.pipelines.DoclingPdfPipeline": 100,
    "legal_agent.scraping.pipelines.MetadataEnrichmentPipeline": 150,
    "legal_agent.scraping.pipelines.ChunkingPipeline": 200,
    "legal_agent.scraping.pipelines.QdrantPipeline": 300,
}

LOG_LEVEL = "INFO"
