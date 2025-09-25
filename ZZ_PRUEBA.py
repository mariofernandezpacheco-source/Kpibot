from utils.logging_enhanced import get_logger, with_correlation_id
logger = get_logger(__name__)

# En cualquier función nueva
with with_correlation_id():
    logger.info("operation_start", ticker="AAPL", component="DAT")