# Missing Values Handling Guide for Terry Stop Dataset

## Summary
- **Total Records**: 65,894
- **Columns with Missing Values**: 2
- **Total Missing Cells**: 33,131 (0.5% of all data)
- **Approach**: Zero-row deletion, smart fill strategy

---

## Missing Values by Column

### 1. Weapon Type
```
Missing Values: 32,565 out of 65,894 (49.42%)
```

**Why It's Missing:**
- Terry Stop form only records weapon type IF a weapon was discovered
- Missing = No weapon was found during the encounter

**Recommended Fill Value:** `'-'`
- Semantic Meaning: "No Weapon" (consistent with existing data)
- Already used in the dataset for this purpose
- Preserves domain knowledge

**Implementation:**
```python
df['Weapon Type'].fillna('-', inplace=True)
```

**Downstream Impact:** ✓ None
- Visualizations already handle '-' values
- Code filters these out explicitly when needed: `df[df['Weapon Type'] != '-']`
- ML models trained on df_clean (pre-cleaning), so no impact

---

### 2. Officer Squad
```
Missing Values: 566 out of 65,894 (0.86%)
```

**Why It's Missing:**
- Data entry gaps for officer squad assignment
- Minimal impact due to small percentage

**Recommended Fill Value:** `'Unknown'`
- Indicates incomplete data
- Allows record inclusion without distortion
- Easy to filter in specific analyses

**Implementation:**
```python
df['Officer Squad'].fillna('Unknown', inplace=True)
```

**Downstream Impact:** ✓ None
- Only 566 records affected (0.86%)
- Analysis code already handles categorical values
- Can be excluded from squad-specific analyses if needed

---

## Integration Strategy

### Option 1: Add to Data Cleaning Cell (Recommended)
Add these lines to Cell 6 after loading df but before creating df_clean:

```python
# Fill missing values - domain-aware approach
df['Weapon Type'].fillna('-', inplace=True)  # '-' = No weapon found
df['Officer Squad'].fillna('Unknown', inplace=True)  # For data entry gaps
```

**Pros:**
- Single point of data cleaning
- Clear data quality documentation
- All downstream analyses use complete data
- Easy to track and modify

### Option 2: Create Separate Preprocessing Cell
Create a dedicated cell between data loading and cleaning:

```python
# Data Quality - Fill Missing Values
# This preserves all 65,894 records while ensuring data completeness

df['Weapon Type'] = df['Weapon Type'].fillna('-')
df['Officer Squad'] = df['Officer Squad'].fillna('Unknown')

print(f"Missing values after filling:")
print(f"  Weapon Type: {df['Weapon Type'].isna().sum()}")
print(f"  Officer Squad: {df['Officer Squad'].isna().sum()}")
```

**Pros:**
- Explicit visibility of preprocessing step
- Easy to enable/disable for testing
- Documentation of data quality decisions

---

## Verification (Run After Implementation)

```python
# Check that missing values are gone
print("Final Data Quality Check:")
print(f"Total missing values: {df.isnull().sum().sum()}")
print(f"Records preserved: {len(df)}")

# Verify fill values are correct
print(f"\nWeapon Type values: {df['Weapon Type'].unique()}")
print(f"Officer Squad sample: {df['Officer Squad'].value_counts().head()}")
```

Expected Output:
```
Final Data Quality Check:
Total missing values: 0
Records preserved: 65894

Weapon Type values: [list including '-' and weapon names]
Officer Squad sample: [counts of squads including 'Unknown']
```

---

## Impact Analysis

| Aspect | Impact | Notes |
|--------|--------|-------|
| **Data Preservation** | ✓ 100% (no records deleted) | All 65,894 records kept |
| **Semantic Correctness** | ✓ Yes | Fills match domain logic |
| **Visualization Output** | ✓ Unchanged | Charts already handle these values |
| **ML Model Training** | ✓ No impact | Models trained on df_clean (after filtering) |
| **Statistical Analysis** | ✓ No change | Aggregations work normally |
| **Filtering Logic** | ✓ Compatible | Code can still filter `!= '-'` if needed |
| **Code Breaking Changes** | ✓ None | All downstream cells remain compatible |

---

## Recommendation Summary

✅ **Implement both fills immediately**
- No risk of breaking existing code
- Improves data quality
- Maintains semantic correctness
- Enables 100% record inclusion
- Simple 2-line addition to data cleaning

✅ **No need to modify downstream cells**
- All visualization code works as-is
- All ML code unaffected (pre-filters anyway)
- All statistical analyses compatible

✅ **Safe approach**
- Zero data loss
- Zero breaking changes
- Zero performance impact
- Easy to revert if needed

---

## Questions to Consider

1. **Are these fill values acceptable for your analysis?**
   - Weapon Type = '-' (No Weapon) → Semantically correct ✓
   - Officer Squad = 'Unknown' → Minimal impact (0.86%) ✓

2. **Should specific analyses exclude these filled values?**
   - Most analyses can use all records as-is
   - Squad-specific analysis can easily filter out 'Unknown'
   - Weapon analyses can work with '-' as "No Weapon" category

3. **Do you want to track data quality over time?**
   - Consider keeping original missing value counts in notebook comments
   - This creates audit trail of data decisions

---

## Quick Implementation Checklist

- [ ] Review fill strategies above
- [ ] Decide on integration option (add to existing cell vs. new cell)
- [ ] Add the 2 lines of fill code
- [ ] Run notebook to verify no errors
- [ ] Confirm output visualizations match previous runs
- [ ] Optional: Add verification code snippet
- [ ] Done! Dataset now 100% complete with 65,894 records
