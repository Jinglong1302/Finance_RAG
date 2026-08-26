"""XBRL tag extractor for SEC filings.

Extracts whitelisted US-GAAP XBRL concepts from SEC filing HTML before
XBRL tags are stripped. Stores concept names, values, and context
references as structured metadata.
"""

from __future__ import annotations

import re
from dataclasses import dataclass

from bs4 import BeautifulSoup, Tag

from src.utils.logging import get_logger

logger = get_logger(__name__)


# Top ~50 standard US-GAAP concepts worth extracting as metadata.
# These cover the most commonly queried financial metrics.
GAAP_WHITELIST: set[str] = {
    # Income Statement
    "us-gaap:Revenues",
    "us-gaap:RevenueFromContractWithCustomerExcludingAssessedTax",
    "us-gaap:CostOfGoodsAndServicesSold",
    "us-gaap:CostOfRevenue",
    "us-gaap:GrossProfit",
    "us-gaap:ResearchAndDevelopmentExpense",
    "us-gaap:SellingGeneralAndAdministrativeExpense",
    "us-gaap:OperatingExpenses",
    "us-gaap:OperatingIncomeLoss",
    "us-gaap:InterestExpense",
    "us-gaap:InterestIncome",
    "us-gaap:IncomeTaxExpenseBenefit",
    "us-gaap:NetIncomeLoss",
    "us-gaap:EarningsPerShareBasic",
    "us-gaap:EarningsPerShareDiluted",
    "us-gaap:WeightedAverageNumberOfSharesOutstandingBasic",
    "us-gaap:WeightedAverageNumberOfDilutedSharesOutstanding",
    # Balance Sheet - Assets
    "us-gaap:Assets",
    "us-gaap:AssetsCurrent",
    "us-gaap:CashAndCashEquivalentsAtCarryingValue",
    "us-gaap:ShortTermInvestments",
    "us-gaap:AccountsReceivableNetCurrent",
    "us-gaap:InventoryNet",
    "us-gaap:PropertyPlantAndEquipmentNet",
    "us-gaap:Goodwill",
    "us-gaap:IntangibleAssetsNetExcludingGoodwill",
    # Balance Sheet - Liabilities
    "us-gaap:Liabilities",
    "us-gaap:LiabilitiesCurrent",
    "us-gaap:AccountsPayableCurrent",
    "us-gaap:LongTermDebt",
    "us-gaap:LongTermDebtNoncurrent",
    # Balance Sheet - Equity
    "us-gaap:StockholdersEquity",
    "us-gaap:RetainedEarningsAccumulatedDeficit",
    "us-gaap:CommonStockSharesOutstanding",
    "us-gaap:TreasuryStockValue",
    # Cash Flow
    "us-gaap:NetCashProvidedByUsedInOperatingActivities",
    "us-gaap:NetCashProvidedByUsedInInvestingActivities",
    "us-gaap:NetCashProvidedByUsedInFinancingActivities",
    "us-gaap:DepreciationDepletionAndAmortization",
    "us-gaap:PaymentsToAcquirePropertyPlantAndEquipment",
    "us-gaap:PaymentsForRepurchaseOfCommonStock",
    "us-gaap:PaymentsOfDividendsCommonStock",
    # Key Ratios / Other
    "us-gaap:CommonStockDividendsPerShareDeclared",
    "us-gaap:ShareBasedCompensation",
    "us-gaap:DeferredRevenueCurrent",
    "us-gaap:OperatingLeaseRightOfUseAsset",
    "us-gaap:RevenueRemainingPerformanceObligation",
}


@dataclass
class XBRLConcept:
    """A single extracted XBRL financial concept."""

    concept: str  # e.g., "us-gaap:Revenues"
    value: str  # e.g., "394328000000"
    context_ref: str  # e.g., "FD2024Q4YTD"
    unit_ref: str  # e.g., "usd"
    decimals: str  # e.g., "-6"


class XBRLExtractor:
    """Extracts whitelisted US-GAAP XBRL concepts from SEC filing HTML.

    Scans <ix:nonFraction> and <ix:nonNumeric> elements for concepts
    matching the GAAP_WHITELIST. Returns structured concept data for
    use as chunk metadata.

    This extractor must be run BEFORE XBRL tags are stripped from the HTML,
    since it reads the XBRL element attributes.
    """

    def __init__(self, whitelist: set[str] | None = None) -> None:
        """Initialize with optional custom whitelist.

        Args:
            whitelist: Set of GAAP concept names to extract.
                      Defaults to GAAP_WHITELIST.
        """
        self.whitelist = whitelist or GAAP_WHITELIST

    def extract(self, html: str) -> list[XBRLConcept]:
        """Extract whitelisted XBRL concepts from filing HTML.

        Args:
            html: Raw SEC filing HTML string (with XBRL tags intact).

        Returns:
            List of extracted XBRLConcept instances.
        """
        soup = BeautifulSoup(html, "lxml")
        concepts: list[XBRLConcept] = []

        # Search for ix:nonFraction elements (numeric values like revenue, EPS)
        for tag in soup.find_all(re.compile(r"ix:nonfraction", re.IGNORECASE)):
            concept = self._extract_from_tag(tag)
            if concept:
                concepts.append(concept)

        # Search for ix:nonNumeric elements (text values)
        for tag in soup.find_all(re.compile(r"ix:nonnumeric", re.IGNORECASE)):
            concept = self._extract_from_tag(tag)
            if concept:
                concepts.append(concept)

        logger.info(f"Extracted {len(concepts)} XBRL concepts from whitelist")
        return concepts

    def _extract_from_tag(self, tag: Tag) -> XBRLConcept | None:
        """Extract concept data from a single XBRL tag if it's in the whitelist.

        Args:
            tag: A BeautifulSoup Tag representing an ix:nonFraction
                 or ix:nonNumeric element.

        Returns:
            XBRLConcept if the tag's name is in the whitelist, else None.
        """
        concept_name = tag.get("name", "")
        if not concept_name:
            return None

        # Check if the concept is in our whitelist
        if concept_name not in self.whitelist:
            return None

        return XBRLConcept(
            concept=concept_name,
            value=tag.get_text(strip=True),
            context_ref=tag.get("contextref", ""),
            unit_ref=tag.get("unitref", ""),
            decimals=tag.get("decimals", ""),
        )

    def get_concept_names_for_chunk(
        self, concepts: list[XBRLConcept], chunk_text: str
    ) -> list[str]:
        """Find which XBRL concepts appear in a given chunk's text.

        Used during chunking to populate the xbrl_concepts metadata field.
        Matches by checking if the concept's value appears in the chunk text.

        Args:
            concepts: All extracted concepts from the filing.
            chunk_text: The text content of a chunk.

        Returns:
            List of concept names found in the chunk text.
        """
        found: list[str] = []
        for concept in concepts:
            if concept.value and concept.value in chunk_text:
                if concept.concept not in found:
                    found.append(concept.concept)
        return found
