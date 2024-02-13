
from fics.dataset_utils import (
    check_dataset,
    get_cik2sic_mapping, 
    get_cik2name_mapping,
    get_sic2desc_mapping,
    get_sic2naics_mapping,
    get_naics2desc_mapping,
    get_cik_stock_mapping,
)

if __name__ == '__main__':
    # check_dataset()
    cik2sic_m = get_cik2sic_mapping()
    print(f'len(cik2sic_m): {len(cik2sic_m)}')
    cik2name_m = get_cik2name_mapping()
    print(f'len(cik2name_m): {len(cik2name_m)}')
    sic2desc_m = get_sic2desc_mapping()
    print(f'len(sic2desc_m): {len(sic2desc_m)}')
    sic2naics_m = get_sic2naics_mapping()
    print(f'len(sic2naics_m): {len(sic2naics_m)}')
    naics2desc_m = get_naics2desc_mapping()
    print(f'len(naics2desc_m): {len(naics2desc_m)}')
    cik_stock_m = get_cik_stock_mapping()
    print(f'len(cik_stock_m): {len(cik_stock_m)}')
