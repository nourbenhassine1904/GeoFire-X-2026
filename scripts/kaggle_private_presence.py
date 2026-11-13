"""Confirm creation from complete owner catalogues; never probe/re-upload rasters."""


def confirm_private_presence(client,owner,slug,*,max_pages=4):
    from kagglesdk.datasets.types.dataset_api_service import ApiListDatasetsRequest
    from kagglesdk.datasets.types.dataset_enums import DatasetSelectionGroup
    views={}
    target=owner+'/'+slug
    target_meta=None
    for name,group in (('mine',DatasetSelectionGroup.DATASET_SELECTION_GROUP_MY),
                       ('private',DatasetSelectionGroup.DATASET_SELECTION_GROUP_MY_PRIVATE),
                       ('public',DatasetSelectionGroup.DATASET_SELECTION_GROUP_MY_PUBLIC)):
        req=ApiListDatasetsRequest();req.group=group;req.page_size=100
        handles=[];tokens=set()
        for _ in range(max_pages):
            page=client.datasets.dataset_api_client.list_datasets(req)
            for item in page.datasets:
                handles.append(item.ref)
                if item.ref==target:
                    target_meta=dict(private=item.is_private,version=item.current_version_number)
            if not page.next_page_token:break
            if page.next_page_token in tokens:raise ValueError('Repeated catalogue pagination token')
            tokens.add(page.next_page_token);req.page_token=page.next_page_token
        else:raise ValueError('Owner catalogue pagination incomplete')
        if len(handles)!=len(set(handles)):raise ValueError('Duplicate catalogue entries')
        views[name]={h for h in handles if h.startswith(owner+'/')}
    # Only the just-created target may temporarily differ while it is indexed.
    stable={name:handles-{target} for name,handles in views.items()}
    if stable['mine']!=stable['private']|stable['public'] or views['private']&views['public']:
        raise ValueError('Owner catalogue views disagree after create; inspect without re-upload')
    if target in views['public'] or target_meta is not None and target_meta['private'] is not True:
        raise ValueError('Private remote presence not confirmed; inspect without re-upload')
    if target not in views['private'] or target not in views['mine'] or target_meta is None or target_meta['version']<1:
        return dict(status='PRIVATE_INDEXING_PENDING',handle=target,
                    basis='Create accepted; target not yet consistently indexed; no upload retry')
    return dict(status='PRIVATE_REMOTE_PRESENCE_CONFIRMED',handle=target,private=True,
                version=target_meta['version'],basis='Complete consistent authenticated owner catalogues',
                GetDataset_required=False)


def wait_private_presence(client,owner,slug,*,checks=8,pause_seconds=15,sleep=None):
    """Bounded catalogue polling for expected indexing, never retry any write."""
    if sleep is None:
        import time
        sleep=time.sleep
    if not 1<=checks<=8:raise ValueError('At most eight indexing checks allowed')
    for index in range(checks):
        result=confirm_private_presence(client,owner,slug)
        if result['status']=='PRIVATE_REMOTE_PRESENCE_CONFIRMED':return result
        if index+1<checks:sleep(pause_seconds)
    raise TimeoutError('Created dataset remains unindexed; preserve state and never re-upload automatically')
